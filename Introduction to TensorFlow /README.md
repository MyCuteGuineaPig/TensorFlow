### Tensorflow

1. **Build model** ``` model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])```
2. **Compile**: specify **optimizer**, **loss**, **metrics** ```model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])```
4. train_generator:
```python
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
        '/tmp/horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # Images may be different size, All images will be resized to 300x300
        batch_size=128       # how many data for one fetching 
        class_mode='binary')
```
5. **Train model**, specify x, y, epoch, callbacks
  - fit: ```model.fit(xs,ys,epochs=500)```
  - fit_generator: specify directory, steps_per_epoch, epochs, verbose
  ```python
    history = model.fit_generator(
        train_generator, # directory
        steps_per_epoch=8, # how many steps for loading data for one epoch
        epochs=15, 
         validation_data=validation_generator, # validation data, can get loss, acc from history 
        verbose=1) # 1 is animation, 0 is no animation
  ```
6. **Test model**: ```test_loss, test_acc  = model.evaluate(test_images, test_labels)```

7. **Get Accuracy and Loss**: 

```
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

### Rule of thumb:

1. **First layer** should be the same as your data(or Error), E.g. 28x28 images, 28 layers of 28 neurons is infeasible. It makes more sense to flatten 28x28 into 784x1 
2. The number of neurons in **last layer** should match the number of classes you are classifying (or Error)

### Some Link to Learn:

[learn more about bias and techniques to avoid it](https://developers.google.com/machine-learning/fairness-overview/)
 
[Fashion MNIST]( https://github.com/zalandoresearch/fashion-mnist)

[```Con2D``` and ```MaxPooling2D```](https://github.com/tensorflow/docs/tree/r1.8/site/en/api_docs) 

[Learn more convolutional 2d layer](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)

[Image Filtering](https://lodev.org/cgtutor/filtering.html) 

[Video of TensorFlow solving real-world problem](https://www.youtube.com/watch?v=NlpS-DhayQA)

[Understanding  Cross Entroy Loss](https://gombru.github.io/2018/05/23/cross_entropy_loss/)

[RMSProp Lecture Slide](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
[RMSProp TensorFlow](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/train/RMSPropOptimizer)
         
[Horses or Humans Convnet](https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb)

[Horses or Humans with Validation](https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%203%20-%20Notebook.ipynb)

[Horses or Humans with Compacting of Images](https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%204%20-%20Notebook.ipynb)

