import numpy as np
import tensorflow as tf
import os


class MyCustomCallback(tf.keras.callbacks.Callback):
  def __init__(self, model_path, test_generator, patience=0):
    super(MyCustomCallback, self).__init__()
    self.patience = patience
    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None
    self.model_path = model_path
    self.test_data = test_generator
 
  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best F1 as 0.0.
    self.best_val_loss = np.inf
    self.is_impatient = False

  def on_train_end(self, logs=None):
    if not self.is_impatient:
      print("Restoring model weights from the end of the best epoch.")
      self.model.set_weights(self.best_weights)
      temp_model_path = self.model_path.replace(".h5", "_temp.h5")
      os.remove(temp_model_path)

  def on_epoch_end(self, epoch, logs=None):
    current_val_loss = logs.get("val_loss")
    print("\n current_val_loss: {}".format(current_val_loss))
    temp_model_path = self.model_path.replace(".h5", "_temp.h5")
    self.model.save(temp_model_path)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}: Testing the model...")
        test_results = self.model.evaluate(self.test_data)
        print(f"Test loss: {test_results[0]}, Test accuracy: {test_results[1]}")

    if current_val_loss < self.best_val_loss:
      self.best_val_loss = current_val_loss
      self.wait = 0
      self.best_weights = self.model.get_weights()
      self.model.save(self.model_path)

    else:
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.is_impatient = True
            self.model.stop_training = True
            print("Restoring model weights from the end of the best epoch.")
            self.model.set_weights(self.best_weights)
            os.remove(temp_model_path)