

import pandas
x_test = pandas.read_pickle("x_test")
y_test = pandas.read_pickle("y_test")
from tensorflow.keras.models import load_model

model = load_model("models/model.h5")

x_train = pandas.read_pickle("x_train")
y_train = pandas.read_pickle("y_train")

score1 = model.evaluate(x_train,y_train,verbose=0)

print("Train loss : ",score1[0])
print("Train accuracy : ",score1[1])

score = model.evaluate(x_test,y_test,verbose=0)

print("Test loss : ",score[0])
print("Test accuracy : ",score[1])




