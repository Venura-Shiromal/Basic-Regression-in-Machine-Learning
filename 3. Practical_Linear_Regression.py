from sklearn.linear_model import SGDRegressor
#from sklearn.preprocessing import StandardScaler
import numpy as np

n = 10000
a = 0.001

x_train, y_train = np.array([1,2,3,4]).reshape(-1,1), np.array([2,4,6,8])

#x_norm = StandardScaler().fit_transform(x_train)

sgdr = SGDRegressor(
    max_iter=n,
    tol=1e-9,
    alpha=0.0,
    eta0=a,
    learning_rate="constant" )

sgdr.fit(x_train, y_train)

#sgdr.fit(x_norm, y_train)

b_norm = sgdr.intercept_
w_norm = sgdr.coef_

w_rounded = np.round(w_norm, 2)
b_rounded = np.round(b_norm, 2)

y_pred_sgd = sgdr.predict(x_train)
y_pred = np.dot(x_train, w_rounded) + b_rounded

#y_pred_sgd = sgdr.predict(x_norm)
#y_pred = np.dot(x_norm, w_norm) + b_norm

print(f"f(x) = {w_rounded}*x + {b_rounded}")


### Commented code is for feature scaling
