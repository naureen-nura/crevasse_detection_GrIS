import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import saving
from tensorflow.keras.metrics import Metric
import numpy as np

@tf.keras.saving.register_keras_serializable(name="mean_iou")
def mean_iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.dtypes.float64)
    y_pred = tf.cast(y_pred, tf.dtypes.float64)
    I = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
    U = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - I
    return tf.reduce_mean(I / U)
    
# @tf.keras.utils.register_keras_serializable(name='mean_iou')
# class mean_iou(tf.keras.metrics.Metric):
#     def __init__(self, name='mean_iou', dtype=tf.float64):
#         super(mean_iou, self).__init__(name=name, dtype=dtype)
#         self.iou = self.add_weight("total", shape=[], initializer='zeros', dtype=self.dtype)

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.cast(y_true, self.dtype)
#         y_pred = tf.cast(y_pred, self.dtype)
#         i = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
#         u = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - i
#         self.iou.assign(tf.reduce_mean(i / u))

#     def result(self):
#         return self.iou

#     def get_config(self):
#         config = super().get_config().copy()

#         config.update({
#              # Return the original input --+
#              #                             |
#              #            +----------------+
#              #            |
#              #            v
#             'mean_iou': self.iou,
#         })

#         return config
        
#     # def get_config(self):
#     #     """Returns the serializable config of the metric."""
#     #     return dict(iou=self.iou,
#     #                 **super(mean_iou,self).get_config())
#         # config = {}
#         # base_config = super(mean_iou, self).get_config()
#         #return dict(list(base_config.items()) + list(config.items()))

#     def reset_state(self):
#         self.iou.assign(0)


# class ConfusionMatrixMetric(tf.keras.metrics.Metric):
#     """A custom Keras metric to compute the running average of the confusion matrix"""
#     def __init__(self, name='confusion_matrix_metric', **kwargs):
#         super(ConfusionMatrixMetric,self).__init__(name='confusion_matrix_metric',**kwargs) # handles base args (e.g., dtype)
#         self.num_classes=num_classes
#         self.total_cm = self.add_weight("total", shape=(num_classes,num_classes), initializer="zeros")
        
#     def reset_state(self):
#         for s in self.variables:
#             s.assign(tf.zeros(shape=s.shape))
            
#     def update_state(self, y_true, y_pred,sample_weight=None):
#         self.total_cm.assign_add(self.confusion_matrix(y_true,y_pred))
#         return self.total_cm
    
#     def get_config(self):
#         """Returns the serializable config of the metric."""
#         config = {}
#         base_config = super(ConfusionMatrixMetric, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
        
#     def result(self):
#         return self.process_confusion_matrix()
    
#     def confusion_matrix(self,y_true, y_pred):
#         """
#         Make a confusion matrix
#         """
#         y_pred=tf.argmax(y_pred,1)
#         cm=tf.math.confusion_matrix(y_true,y_pred,dtype=tf.float32,num_classes=self.num_classes)
#         return cm
    
#     def process_confusion_matrix(self):
#         "returns precision, recall and f1 along with overall accuracy"
#         cm=self.total_cm
#         diag_part=tf.linalg.diag_part(cm)
#         precision=diag_part/(tf.reduce_sum(cm,0)+tf.constant(1e-15))
#         recall=diag_part/(tf.reduce_sum(cm,1)+tf.constant(1e-15))
#         f1=2*precision*recall/(precision+recall+tf.constant(1e-15))
#         return precision,recall,f1
    
#     def fill_output(self,output):
#         results=self.result()
#         for i in range(self.num_classes):
#             output['precision_{}'.format(i)]=results[0][i]
#             output['recall_{}'.format(i)]=results[1][i]
#            output['F1_{}'.format(i)]=results[2][i]
    
@tf.keras.saving.register_keras_serializable(name="dice_coefficient")
def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice