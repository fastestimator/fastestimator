import tensorflow as tf
import torch

from fastestimator.backend.reduce_loss import reduce_loss


def update_model(model, loss, tape=None):
    loss = reduce_loss(loss)
    if isinstance(model, tf.keras.Model):
        with tape.stop_recording():
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    elif isinstance(model, torch.nn.Module):
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
    else:
        raise ValueError("Unrecognized model instance {}".format(type(model)))
