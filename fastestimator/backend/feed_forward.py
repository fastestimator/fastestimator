import torch
import tensorflow as tf

def feed_forward(model, x, training=True):
    if isinstance(model, tf.keras.Model):
        x = model(x, training=training)
    elif isinstance(model, torch.nn.Module):
        model.train(mode=training)
        x = model(x)
    else:
        raise ValueError("Unrecognized model instance {}".format(type(model)))
    return x