# import pickle 
# x_test = pickle.load(open("x_test","rb"))
# y_test = pickle.load(open("y_test","rb"))

import pandas
x_test = pandas.read_pickle("x_test")
y_test = pandas.read_pickle("y_test")
from tensorflow.keras.models import load_model

model = load_model("models/model.h5")

# y_predict = model.predict(x_test)


score = model.evaluate(x_test,y_test)

print("Test loss : ",score[0])
print("test accuracy : ",score[1])




