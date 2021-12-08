import numpy as np
import numpy.random as rnd
import sklearn
import matplotlib.pyplot as plt

np.random.seed(42)

N = 100
x = np.random.rand(N, 1) 
t=np.sin(2*np.pi*x)+0.2*rnd.randn(N,1)

x1=np.linspace(0,1,100)
plt.plot(x, t, "bo")
plt.plot(x1,np.sin(2*np.pi*x1),'g')
plt.xlabel("x", fontsize=18)
plt.ylabel("y", rotation=0, fontsize=18)
plt.axis([0, 1, -1.5, 1.5])
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(x)


lin_reg = LinearRegression()
lin_reg.fit(x_poly, t)
lin_reg.intercept_, lin_reg.coef_

x_new=np.linspace(0, 1, 100).reshape(100, 1)
x_new_poly = poly_features.transform(x_new)
t_new = lin_reg.predict(x_new_poly)
plt.plot(x, t, "bo")
plt.plot(x_new, t_new, "r")
plt.plot(x_new,np.sin(2*np.pi*x_new),'g')
plt.xlabel("x", fontsize=18)
plt.ylabel("t", rotation=0, fontsize=18)
plt.axis([0, 1, -1.5, 1.5])
plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   
    plt.xlabel("Training set size", fontsize=14) 
    plt.ylabel("RMSE", fontsize=14)              

N = 100
x = np.random.rand(N, 1) 
y=np.sin(2*np.pi*x)+0.2*rnd.randn(N,1)

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, x, y)
plt.axis([0, 80, 0, 3])                         
plt.show()                                      

from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

plot_learning_curves(polynomial_regression, x, y)
plt.axis([0, 80, 0, 3])           
plt.show()                        

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

ridge_reg=Ridge(alpha=1e-5)
model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                    ("regul_reg", ridge_reg),
                 ])
model.fit(x,t)
x1=np.linspace(0,1,100).reshape(100,1)
y1=model.predict(x1)
plt.plot(x, t, "bo")
plt.plot(x1,y1,'g')
plt.xlabel("x", fontsize=18)
plt.ylabel("y", rotation=0, fontsize=18)
plt.axis([0, 1, -1.5, 1.5])
plt.show()