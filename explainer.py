import shap 

from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler
x_train = pickle.load(open("x_train","rb"))
x_test = pickle.load(open("x_test","rb"))
model = load_model('models/model.h5')

# scale = StandardScaler()
# x_train = scale.fit_transform(x_train)
# x_test = scale.fit_transform(x_test)
# explainer = shap.DeepExplainer(model,x_train)
# shap_values = explainer.shap_values(x_train)

shap.initjs()
explainer_shap = shap.DeepExplainer(model=model,data = x_train)


# print("ans")
# print(x_train)
# shap.summary_plot(shap_values[0],plot_type="bar",feature_names=x_test.columns)

# shap.plots.bar(shap_values)
