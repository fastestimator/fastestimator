import numpy as np
import tensorflow as tf
from fastestimator.architecture.retinanet import RetinaNet, get_fpn_anchor_box, get_target, PredictBox
from fastestimator.dataset import svhn_data
from fastestimator.estimator.estimator import Estimator
from fastestimator.network.loss import Loss
from fastestimator.network.model import ModelOp, build
from fastestimator.network.network import Network
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.pipeline.preprocess import Minmax
from fastestimator.record.preprocess import ImageReader, Resize
from fastestimator.record.record import RecordWriter
from fastestimator.util.op import NumpyOp, TensorOp

class String2List(NumpyOp):
    #this thing converts '[1, 2, 3]' into np.array([1, 2, 3])
    def forward(self, data):
        for idx, elem in enumerate(data):
            data[idx] =  np.array([int(x) for x in elem[1:-1].split(',')])
        return data

class RelativeCoordinate(NumpyOp):
    def forward(self, data):
        image, x1, y1, x2, y2 = data
        height, width = image.shape[0], image.shape[1]
        x1, y1, x2, y2 = x1/width, y1/height, x2/width, y2/height
        return (x1, y1, x2, y2)

class GenerateTarget(NumpyOp):
    def __init__(self, inputs=None, outputs=None, mode=None):
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode
        self.anchorbox = get_fpn_anchor_box(input_shape=(64, 128, 3))

    def forward(self, data):
        label, x1, y1, x2, y2 = data
        target_cls, target_loc = get_target(self.anchorbox, label, x1, y1, x2, y2, num_classes=10)
        return target_cls,target_loc

class RetinaLoss(Loss):
    def focal_loss(self, cls_gt, cls_pred, num_classes, alpha=0.25, gamma=2.0):
        #cls_gt has shape [B, A], cls_pred is in [B, A, K]
        obj_idx = tf.where(tf.greater_equal(cls_gt, 0)) #index of object
        obj_bg_idx = tf.where(tf.greater_equal(cls_gt, -1)) #index of object and background
        cls_gt = tf.one_hot(cls_gt, num_classes)
        cls_gt = tf.gather_nd(cls_gt, obj_bg_idx)
        cls_pred = tf.gather_nd(cls_pred, obj_bg_idx)
        #getting the object count for each image in batch
        _, idx, count = tf.unique_with_counts(obj_bg_idx[:,0])
        object_count = tf.gather_nd(count, tf.reshape(idx, (-1, 1)))
        object_count = tf.tile(tf.reshape(object_count,(-1, 1)), [1,num_classes])
        object_count = tf.cast(object_count, tf.float32)
        #reshape to the correct shape
        cls_gt = tf.reshape(cls_gt, (-1, 1))
        cls_pred = tf.reshape(cls_pred, (-1, 1))
        object_count = tf.reshape(object_count, (-1, 1))
        # compute the focal weight on each selected anchor box
        alpha_factor = tf.ones_like(cls_gt) * alpha
        alpha_factor = tf.where(tf.equal(cls_gt, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(tf.equal(cls_gt, 1), 1 - cls_pred, cls_pred)
        focal_weight = alpha_factor * focal_weight ** gamma / object_count
        focal_loss =  tf.losses.BinaryCrossentropy()(cls_gt, cls_pred, sample_weight=focal_weight)
        return focal_loss, obj_idx

    def smooth_l1(self, loc_gt, loc_pred, obj_idx):
        #loc_gt anf loc_pred has shape [B, A, 4]
        loc_gt = tf.gather_nd(loc_gt, obj_idx)
        loc_pred = tf.gather_nd(loc_pred, obj_idx)
        loc_gt = tf.reshape(loc_gt, (-1, 1))
        loc_pred = tf.reshape(loc_pred, (-1, 1))
        loc_diff = tf.abs(loc_gt - loc_pred)
        smooth_l1_loss = tf.where(tf.less(loc_diff,1), 0.5 * loc_diff**2, loc_diff-0.5)
        smooth_l1_loss = tf.reduce_mean(smooth_l1_loss)
        return smooth_l1_loss
    
    def calculate_loss(self, batch, prediction):
        cls_gt, loc_gt = batch["target_cls"], batch["target_loc"]
        cls_pred, loc_pred = prediction["pred_cls"], prediction["pred_loc"]
        focal_loss, obj_idx = self.focal_loss(cls_gt, cls_pred, num_classes=10)
        smooth_l1_loss = self.smooth_l1(loc_gt, loc_pred, obj_idx)
        return 40000*focal_loss+smooth_l1_loss

def get_estimator():
    #prepare data in disk
    train_csv, val_csv, path = svhn_data.load_data()
    writer = RecordWriter(train_data=train_csv,
                          validation_data=val_csv,
                          ops=[ImageReader(inputs="image", parent_path=path, outputs="image"), 
                               String2List(inputs=["label", "x1", "y1", "x2", "y2"], outputs=["label", "x1", "y1", "x2", "y2"]),
                               RelativeCoordinate(inputs=("image", "x1", "y1", "x2", "y2"), outputs=("x1", "y1", "x2", "y2")),
                               Resize(inputs="image", target_size=(64, 128),outputs="image"),
                               GenerateTarget(inputs=("label", "x1", "y1", "x2", "y2"), outputs=("target_cls", "target_loc"))])
    #prepare pipeline
    pipeline = Pipeline(batch_size=256,
                        data=writer,
                        ops=Minmax(inputs="image", outputs="image"),
                        read_feature=["image", "target_cls", "target_loc"])
    #prepare model
    model = build(keras_model=RetinaNet(input_shape=(64, 128, 3), num_classes=10),
                  loss=RetinaLoss(),
                  optimizer=tf.optimizers.Adam(learning_rate=0.0001))
    network = Network(ops=[ModelOp(inputs="image", model=model, outputs=["pred_cls", "pred_loc"]),
                           PredictBox(top_n=10, score_threshold=0.2, outputs=("cls_selected", "loc_selected", "valid_outputs"), mode="eval")])
    #prepare estimator
    estimator = Estimator(network= network,
                          pipeline=pipeline,
                          epochs= 15,
                          log_steps=1,
                          steps_per_epoch=1,)
    return estimator