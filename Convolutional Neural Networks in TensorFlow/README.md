Augmentaion 

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



### Link of Resources

[famous Kaggle Dogs v Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/overview/winners)

[Image Augmentation](https://github.com/keras-team/keras-preprocessing): like rotating 90 degrees of images. It doesn't require you to edit your raw images, nor does it amend them for you on-disk. It does it in-memory as it's performing the training, allowing you to experiment without impacting your dataset.

Image Augmentation implementation: [APIs at the Keras](https://keras.io/preprocessing/image/)
