import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

excel_data = pd.read_excel('adult.xlsx')
print(excel_data.head())
label = LabelEncoder()
dicts = {}

label.fit(excel_data.workclass.drop_duplicates())
dicts['workclass'] = list(label.classes_)
excel_data.workclass = label.transform(excel_data.workclass)

label.fit(excel_data['native-country'].drop_duplicates())
dicts['native-country'] = list(label.classes_)
excel_data['native-country'] = label.transform(excel_data['native-country'])

label.fit(excel_data['y'].drop_duplicates())
dicts['y'] = list(label.classes_)
excel_data['y'] = label.transform(excel_data['y'])

label.fit(excel_data['occupation'].drop_duplicates())
dicts['occupation'] = list(label.classes_)
excel_data['occupation'] = label.transform(excel_data['occupation'])

label.fit(excel_data['marital-status'].drop_duplicates())
dicts['marital-status'] = list(label.classes_)
excel_data['marital-status'] = label.transform(excel_data['marital-status'])

label.fit(excel_data['relationship'].drop_duplicates())
dicts['relationship'] = list(label.classes_)
excel_data['relationship'] = label.transform(excel_data['relationship'])

label.fit(excel_data['race'].drop_duplicates())
dicts['race'] = list(label.classes_)
excel_data['race'] = label.transform(excel_data['race'])

label.fit(excel_data['sex'].drop_duplicates())
dicts['sex'] = list(label.classes_)
excel_data['sex'] = label.transform(excel_data['sex'])

label.fit(excel_data['education'].drop_duplicates())
dicts['education'] = list(label.classes_)
excel_data['education'] = label.transform(excel_data['education'])

print(excel_data.head())

target = excel_data.y
train = excel_data.drop(['y'], axis=1)
print(train.head())

x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.5)
model_linear = LinearRegression()
model_rfc = RandomForestClassifier(n_estimators=70)
model_knc = KNeighborsClassifier(n_neighbors=18)
model_rfc.fit(x_train, y_train)
model_linear.fit(x_train, y_train)
model_knc.fit(x_train, y_train)
model_pred_rfc = model_rfc.predict(x_test)
model_pred_knc = model_knc.predict(x_test)
model_pred_linear = model_linear.predict(x_test)
print('*******************************')
print('Random Forest')
print(model_pred_rfc, '***', y_test)
print(np.mean(abs(y_test - model_pred_rfc)))
print('*******************************')
print('KNeighboursPredictions')
print(model_pred_knc, '***', y_test)
print(np.mean(abs(y_test - model_pred_knc)))
print('*******************************')
print('LinearPredictions')
print(model_pred_linear, '***', y_test)
print(np.mean(abs(y_test - model_pred_linear)))
