import os
import tempfile
from ast import literal_eval

import fastestimator as fe
import tensorflow as tf
from fastestimator.architecture.retinanet import RetinaNet, get_fpn_anchor_box, get_target
from fastestimator.dataset.mscoco import load_data
from fastestimator.op import NumpyOp
from fastestimator.op.numpyop import ImageReader, ResizeImageAndBbox, TypeConverter
from fastestimator.op.tensorop import Loss, ModelOp, Pad, Rescale
from fastestimator.trace import ModelSaver


class String2List(NumpyOp):
    # this thing converts '[1, 2, 3]' into np.array([1, 2, 3])
    def forward(self, data, state):
        data = map(literal_eval, data)
        return data


class GenerateTarget(NumpyOp):
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.anchorbox = get_fpn_anchor_box(input_shape=(512, 512, 3))

    def forward(self, data, state):
        obj_label, x1, y1, width, height = data
        cls_gt, x1_gt, y1_gt, w_gt, h_gt = get_target(self.anchorbox, obj_label, x1, y1, width, height)
        return cls_gt, x1_gt, y1_gt, w_gt, h_gt


class RetinaLoss(Loss):
    def focal_loss(self, cls_gt_example, cls_pred_example, alpha=0.25, gamma=2.0):
        # cls_gt_example shape: [A], cls_pred_example shape: [A, K]
        num_classes = cls_pred_example.shape[-1]
        # gather the objects and background, discard the rest
        anchor_obj_idx = tf.where(tf.greater_equal(cls_gt_example, 0))
        obj_bg_idx = tf.where(tf.greater_equal(cls_gt_example, -1))
        anchor_obj_count = tf.cast(tf.shape(anchor_obj_idx)[0], tf.float32)
        cls_gt_example = tf.one_hot(cls_gt_example, num_classes)
        cls_gt_example = tf.gather_nd(cls_gt_example, obj_bg_idx)
        cls_pred_example = tf.gather_nd(cls_pred_example, obj_bg_idx)
        cls_gt_example = tf.reshape(cls_gt_example, (-1, 1))
        cls_pred_example = tf.reshape(cls_pred_example, (-1, 1))
        # compute the focal weight on each selected anchor box
        alpha_factor = tf.ones_like(cls_gt_example) * alpha
        alpha_factor = tf.where(tf.equal(cls_gt_example, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(tf.equal(cls_gt_example, 1), 1 - cls_pred_example, cls_pred_example)
        focal_weight = alpha_factor * focal_weight**gamma / anchor_obj_count
        cls_loss = tf.losses.BinaryCrossentropy(reduction='sum')(cls_gt_example,
                                                                 cls_pred_example,
                                                                 sample_weight=focal_weight)
        return cls_loss, anchor_obj_idx

    def smooth_l1(self, loc_gt_example, loc_pred_example, anchor_obj_idx):
        # loc_gt is padded x 4, loc_pred is #num_anchor x 4
        loc_pred = tf.gather_nd(loc_pred_example, anchor_obj_idx)  #anchor_obj_count x 4
        anchor_obj_count = tf.shape(loc_pred)[0]
        loc_gt = loc_gt_example[:anchor_obj_count]  #anchor_obj_count x 4
        loc_gt = tf.reshape(loc_gt, (-1, 1))
        loc_pred = tf.reshape(loc_pred, (-1, 1))
        loc_diff = tf.abs(loc_gt - loc_pred)
        smooth_l1_loss = tf.where(tf.less(loc_diff, 1), 0.5 * loc_diff**2, loc_diff - 0.5)
        smooth_l1_loss = tf.reduce_sum(smooth_l1_loss) / tf.cast(anchor_obj_count, tf.float32)
        return smooth_l1_loss

    def forward(self, data, state):
        cls_gt, x1_gt, y1_gt, w_gt, h_gt, pred_cls, pred_loc = data
        local_batch_size = state["local_batch_size"]
        focal_loss = []
        l1_loss = []
        total_loss = []
        for idx in range(local_batch_size):
            cls_gt_example = cls_gt[idx]
            x1_gt_example = x1_gt[idx]
            y1_gt_example = y1_gt[idx]
            w_gt_example = w_gt[idx]
            h_gt_example = h_gt[idx]
            loc_gt_example = tf.transpose(tf.stack([x1_gt_example, y1_gt_example, w_gt_example, h_gt_example]))
            cls_pred_example = pred_cls[idx]
            loc_pred_example = pred_loc[idx]
            focal_loss_example, anchor_obj_idx = self.focal_loss(cls_gt_example, cls_pred_example)
            smooth_l1_loss_example = self.smooth_l1(loc_gt_example, loc_pred_example, anchor_obj_idx)
            focal_loss.append(focal_loss_example)
            l1_loss.append(smooth_l1_loss_example)
        focal_loss = tf.stack(focal_loss)
        l1_loss = tf.stack(l1_loss)
        total_loss = focal_loss + l1_loss
        return total_loss


def get_estimator(data_path=None, model_dir=tempfile.mkdtemp()):
    #prepare dataset
    train_csv, val_csv, path = load_data(path=data_path)
    writer = fe.RecordWriter(
        save_dir=os.path.join(path, "retinanet_coco"),
        train_data=train_csv,
        validation_data=val_csv,
        ops=[
            ImageReader(inputs="image", parent_path=path, outputs="image"),
            String2List(inputs=["x1", "y1", "width", "height", "obj_label"],
                        outputs=["x1", "y1", "width", "height", "obj_label"]),
            ResizeImageAndBbox(target_size=(512, 512),
                               keep_ratio=True,
                               inputs=["image", "x1", "y1", "width", "height"],
                               outputs=["image", "x1", "y1", "width", "height"]),
            GenerateTarget(inputs=("obj_label", "x1", "y1", "width", "height"),
                           outputs=("cls_gt", "x1_gt", "y1_gt", "w_gt", "h_gt")),
            TypeConverter(target_type='int32',
                          inputs=["num_obj", "x1", "y1", "width", "height", "obj_label", "cls_gt"],
                          outputs=["num_obj", "x1", "y1", "width", "height", "obj_label", "cls_gt"]),
            TypeConverter(target_type='float32',
                          inputs=["x1_gt", "y1_gt", "w_gt", "h_gt"],
                          outputs=["x1_gt", "y1_gt", "w_gt", "h_gt"])
        ],
        compression="GZIP",
        write_feature=["image", "cls_gt", "x1_gt", "y1_gt", "w_gt", "h_gt"])
    # prepare pipeline
    pipeline = fe.Pipeline(
        batch_size=4,
        data=writer,
        ops=[
            Rescale(inputs="image", outputs="image"),
            Pad(padded_shape=[252],
                inputs=["x1_gt", "y1_gt", "w_gt", "h_gt"],
                outputs=["x1_gt", "y1_gt", "w_gt", "h_gt"])
        ])
    # prepare network
    model = fe.build(model_def=lambda: RetinaNet(input_shape=(512, 512, 3), num_classes=90),
                     model_name="retinanet",
                     optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                     loss_name="total_loss")
    network = fe.Network(ops=[
        ModelOp(inputs="image", model=model, outputs=["pred_cls", "pred_loc"]),
        RetinaLoss(inputs=("cls_gt", "x1_gt", "y1_gt", "w_gt", "h_gt", "pred_cls", "pred_loc"), outputs="total_loss")
    ])
    # prepare estimator
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=13,
                             traces=ModelSaver(model_name="retinanet", save_dir=model_dir, save_best=True))
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
