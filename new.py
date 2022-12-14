

import pandas
x_test = pandas.read_pickle("x_test")
y_test = pandas.read_pickle("y_test")
from tensorflow.keras.models import load_model

model = load_model("models/model.h5")



score = model.evaluate(x_test,y_test,verbose=0)

print("Test loss : ",score[0])
print("Test accuracy : ",score[1])




