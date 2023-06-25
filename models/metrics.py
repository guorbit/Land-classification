"""
Package for custom metrics
"""
import keras
import tensorflow as tf
from sklearn.metrics import f1_score


class LossMetric(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred):
        # Implement the logic to update the metric state
        pass

    def result(self):
        # Return the metric value
        pass

class F1Metric(keras.metrics.Metric):
    """
    F1 metric for keras model
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred):
        self.f1 = f1_score(y_true, y_pred, average='macro')

    def result(self):
        return self.f1
    