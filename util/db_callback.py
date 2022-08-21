
import keras
from evaluate_results import log_epoch_end_performances



class LossHistory(keras.callbacksCallback):
    def on_epoch_end(self, epoch, logs=None):
        log_epoch_end_performances(epoch=epoch, h=self.model, encoded_validate_x=self.model.validation_data)