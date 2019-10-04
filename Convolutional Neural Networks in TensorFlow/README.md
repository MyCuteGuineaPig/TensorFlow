#### Augmentaion 

```python
train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range = 40, #image  will rotate between 0 - 40 degree
    width_shift_range = 0.2,  # randomly shift up to 20% horizontal 
    height_shift_range = 0.2, # randomly shift up to 20% vertically
    shear_range = 0.2, #shear up randomly 20% percent of the image
    zoom_range =0.2, #zoom random amount up to 20% of image
    horizontal_flip = True, # image will be flip randomly 
    fill_mode = 'nearest' # fill nay pixels that maybe lost by the operations. I'm just going to stick with nearest here.
    #with the neighbors of that pixels to try and keep uniformity
)

```
#### Transfer Learning 

- use Inception
- lock pretrained layers

```python

from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150,150,3), 
                                include_top = False, # specify that ignore fully-connected layer at the top and straight to the convolution
                                weights = None) # you don't want to use the built-in weights but use the snapshot you just download it

pre_trained_model.load_weights(local_weights_file)


#****************************************# 
# Iterate through Layers  to lock layers without changing parameters
#****************************************# 

for layer in pre_trained_model.layers:  
  layer.trainable = False # specify not to train within the local

pre_trained_model.summary() #huge code
```

Using Dropout 

```python
last_layer = pre_trained_model.get_layer('mixed7') 
last_output = last_layer.output # which is a output of a lot of convolution that are 7 by7

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation = 'relu')(x)
x = layers.Dropout(0.2)(x) # Parameter is between 0 and 1, it's fraction unit to drop
x = layers.Dense(1, activation = 'sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizers = RMSprop(lr = 0.001), 
              loss = 'binary_crossentropy', 
              metrics = ['acc'])
```





### Link of Resources

[famous Kaggle Dogs v Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/overview/winners)

#### Augmentation

[Image Augmentation](https://github.com/keras-team/keras-preprocessing): like rotating 90 degrees of images. It doesn't require you to edit your raw images, nor does it amend them for you on-disk. It does it in-memory as it's performing the training, allowing you to experiment without impacting your dataset.

[APIs at the Keras](https://keras.io/preprocessing/image/):  Image Augmentation implementation

#### Transfer Learning

[TensorFlow Transfer learning with a pretrained ConvNet](https://www.tensorflow.org/tutorials/images/transfer_learning)

[A copy of pretrained weights for inception neural network] (https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels)

[Understanding Dropout] (https://www.youtube.com/watch?v=ARq74QuavAo)


