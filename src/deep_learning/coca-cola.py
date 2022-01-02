# Save Model Using joblib
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
url = "dpl_proj/data/COCO COLA.csv"
names = ['Date', 'Open', 'High', 'Low',	'Close', 'Adj Close', 'Volume']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[1:,1:8]
# print('X')
# print(X)
Y = array[1:,6]
# print('Y')
# print(Y)
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on training set
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_cc_model.sav'
joblib.dump(model, filename)
 
# some time later...
 
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)
