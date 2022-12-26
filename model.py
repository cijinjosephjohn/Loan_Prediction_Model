import tensorflow as tf

import pandas

x_train = pandas.read_pickle("x_train")
y_train = pandas.read_pickle("y_train")



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=4,activation="relu"))
model.add(tf.keras.layers.Dense(units=4,activation="relu"))
model.add(tf.keras.layers.Dense(units=4,activation="relu"))
model.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])


model.fit(x_train,y_train,epochs=13)
import os 
model.save(os.path.join("models","model.h5"))

