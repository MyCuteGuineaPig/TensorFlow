1. **Build model** ``` model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])```
2. **Compile**: specify **optimizer** and **loss** ```model.compile(optimizer='sgd', loss='mean_squared_error')```
3. **Train model**, specify x, y, epoch, callbacks
  - Fit: ```model.fit(xs,ys,epochs=500)```
