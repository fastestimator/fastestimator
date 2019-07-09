from fastestimator.pipeline.dynamic.preprocess import AbstractPreprocessing as AbstractPreprocessingD
from fastestimator.architecture.retinanet import RetinaNet, get_fpn_anchor_box, get_target
from fastestimator.pipeline.dynamic.preprocess import ImageReader
from fastestimator.pipeline.static.preprocess import Minmax
from fastestimator.estimator.estimator import Estimator
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.estimator.trace import Accuracy, Trace
from fastestimator.dataset import svhn_data
import tensorflow as tf
import numpy as np
import cv2

class Network:
    def __init__(self):
        self.model = RetinaNet(input_shape=(64, 128, 3), num_classes=10)
        self.optimizer = tf.optimizers.Adam(learning_rate=0.0001)
        self.loss = MyLoss()
        self.anchorbox = tf.convert_to_tensor(get_fpn_anchor_box(input_shape=(64, 128, 3)))
        self.anchor_w_h = tf.tile(self.anchorbox[:,2:], [1, 2]) - tf.tile(self.anchorbox[:, :2], [1, 2])

    def train_op(self, batch):
        with tf.GradientTape() as tape:
            predictions = self.model(batch["image"])
            loss = self.loss((batch["target_cls"], batch["target_loc"]), predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return predictions, loss

    def eval_op(self, batch):
        top_n = 10
        score_threshold = 0.2
        predictions = self.model(batch["image"], training=False)
        loss = self.loss((batch["target_cls"], batch["target_loc"]), predictions)
        cls_pred, loc_pred = tuple(predictions)
        #convert the residual prediction to absolute prediction in (x1, y1, x2, y2)
        loc_pred = tf.map_fn(lambda x: x * self.anchor_w_h + self.anchorbox, elems=loc_pred, dtype=tf.float32, back_prop=False)
        num_batch, num_anchor, _ = loc_pred.shape
        cls_best_score = tf.reduce_max(cls_pred, axis=-1)
        cls_best_class = tf.argmax(cls_pred, axis=-1)
        #select top n anchor boxes to proceed 
        sorted_score = tf.sort(cls_best_score,  direction='DESCENDING')
        top_n = tf.minimum(top_n, num_anchor)
        cls_best_score = tf.cond(tf.greater(num_anchor, top_n),
                                lambda: tf.where(tf.greater_equal(cls_best_score, tf.tile(sorted_score[:,top_n-1:top_n],[1, num_anchor])), cls_best_score, 0.0),
                                lambda: cls_best_score)
        #Padded Nonmax suppression with threshold
        selected_indices_padded = tf.map_fn(lambda x: tf.image.non_max_suppression_padded(x[0], x[1], top_n, pad_to_max_output_size=True, score_threshold=score_threshold).selected_indices, (loc_pred, cls_best_score), dtype=tf.int32, back_prop=False)
        valid_outputs = tf.map_fn(lambda x: tf.image.non_max_suppression_padded(x[0], x[1], top_n, pad_to_max_output_size=True, score_threshold=score_threshold).valid_outputs, (loc_pred, cls_best_score), dtype=tf.int32, back_prop=False)
        #select output anchors after the NMS
        batch_index = tf.tile(tf.reshape(tf.range(num_batch),[-1, 1]), [1, top_n])
        selected_indices_padded = tf.stack([batch_index, selected_indices_padded], axis=-1)
        select_mask = tf.sequence_mask(valid_outputs, top_n)
        selected_anchors = tf.boolean_mask(selected_indices_padded, select_mask)
        #get the class and coordinates or output anchor
        loc_selected = tf.gather_nd(loc_pred, selected_anchors)
        cls_selected = tf.gather_nd(cls_best_class, selected_anchors)
        output = (cls_selected, loc_selected, valid_outputs)
        return output, loss

class MyPipeline(Pipeline):
    def edit_feature(self, feature):
        height, width = feature["image"].shape[0], feature["image"].shape[1]
        feature["x1"], feature["y1"], feature["x2"], feature["y2"] = feature["x1"]/width, feature["y1"]/height, feature["x2"]/width, feature["y2"]/height
        feature["image"] = cv2.resize(feature["image"], (128, 64)) #cv2 requires (width, height)
        anchorbox = get_fpn_anchor_box(input_shape=feature["image"].shape)
        target_cls, target_loc = get_target(anchorbox, feature["label"], feature["x1"], feature["y1"], feature["x2"], feature["y2"], num_classes=10)
        feature["target_cls"], feature["target_loc"] = target_cls, target_loc
        return feature

class String2List(AbstractPreprocessingD):
    #this thing converts '[1, 2, 3]' into np.array([1, 2, 3])
    def transform(self, data):
        data = np.array([int(x) for x in data[1:-1].split(',')])
        return data

class MyLoss(tf.losses.Loss):
    def call(self, y_true, y_pred):
        cls_gt, loc_gt = tuple(y_true)
        cls_pred, loc_pred = tuple(y_pred)
        focal_loss, obj_idx = self.focal_loss(cls_gt, cls_pred, num_classes=10)
        smooth_l1_loss = self.smooth_l1(loc_gt, loc_pred, obj_idx)
        return 40000*focal_loss+smooth_l1_loss

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

def get_estimator():
    train_csv, test_csv, path = svhn_data.load_data()

    pipeline = MyPipeline(batch_size=256,
                          feature_name=["image", "label", "x1", "y1", "x2", "y2", "target_cls", "target_loc"],
                          train_data=train_csv,
                          validation_data=test_csv,
                          transform_dataset=[[ImageReader(parent_path=path)], [String2List()], [String2List()], [String2List()], [String2List()], [String2List()], [],[]],
                          transform_train= [[Minmax()], [], [], [],[],[],[],[]],
                          padded_batch=True)
    
    estimator = Estimator(network= Network(),
                          pipeline=pipeline,
                          epochs= 15,
                          log_steps=20)
    return estimator
