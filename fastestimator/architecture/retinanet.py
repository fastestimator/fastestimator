from tensorflow.keras import layers, models
import tensorflow as tf

def classification_sub_net(num_classes, num_anchor=9):
    model = models.Sequential()
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(num_classes * num_anchor,  kernel_size=3, strides=1, padding='same', activation='sigmoid'))
    model.add(layers.Reshape((-1, num_classes)))  #the output dimension is [batch, #anchor, #classes]
    return model

def regression_sub_net(num_anchor=9):
    model = models.Sequential()
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(256,  kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(4 * num_anchor,  kernel_size=3, strides=1, padding='same', activation='sigmoid'))
    model.add(layers.Reshape((-1, 4)))  #the output dimension is [batch, #anchor, 4]
    return model

def RetinaNet(input_shape, num_classes, num_anchor=9):
    inputs = tf.keras.Input(shape= input_shape)
    #FPN
    resnet50 = tf.keras.applications.ResNet50(weights= "imagenet", include_top= False, input_tensor=inputs, pooling=None)
    assert resnet50.layers[80].name == "activation_21"
    C3 = resnet50.layers[80].output
    assert resnet50.layers[142].name == "activation_39"
    C4 = resnet50.layers[142].output
    assert resnet50.layers[-1].name == "activation_48"
    C5 = resnet50.layers[-1].output
    P5 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same')(C5)
    P5_upsampling = layers.UpSampling2D()(P5)
    P4 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same')(C4)
    P4 = layers.Add()([P5_upsampling, P4])
    P4_upsampling = layers.UpSampling2D()(P4)
    P3 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same')(C3)
    P3 = layers.Add()([P4_upsampling, P3])
    P6 = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(C5)
    P7 = layers.Activation('relu')(P6)
    P7 = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(P7)
    P5 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(P5)
    P4 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(P4)
    P3 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(P3)
    #classification subnet
    cls_subnet = classification_sub_net(num_classes=num_classes, num_anchor=num_anchor)
    P3_cls = cls_subnet(P3)
    P4_cls = cls_subnet(P4)
    P5_cls = cls_subnet(P5)
    P6_cls = cls_subnet(P6)
    P7_cls = cls_subnet(P7)
    cls_output = layers.Concatenate(axis=-2)([P3_cls, P4_cls, P5_cls, P6_cls, P7_cls])
    #localization subnet
    loc_subnet = regression_sub_net(num_anchor=num_anchor)
    P3_loc = loc_subnet(P3)
    P4_loc = loc_subnet(P4)
    P5_loc = loc_subnet(P5)
    P6_loc = loc_subnet(P6)
    P7_loc = loc_subnet(P7)
    loc_output = layers.Concatenate(axis=-2)([P3_loc, P4_loc, P5_loc, P6_loc, P7_loc])
    return tf.keras.Model(inputs=inputs, outputs=[cls_output, loc_output]) 


