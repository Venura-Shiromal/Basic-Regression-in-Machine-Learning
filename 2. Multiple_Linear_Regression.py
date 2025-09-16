import numpy as np

#### FUNCTIONS ####
def derive(x, y, m, w, b):
    p = len(x[0])   # number of features
    dj_w = np.zeros(p)
    dj_b = 0

    for i in range(m):
        err = (np.dot(w, x[i]) + b - y[i])
        for j in range(p):
            dj_w[j] += err * x[i][j]
        dj_b += err

    dj_w /= m
    dj_b /= m
    return dj_w, dj_b

def gradient_descent(a, x, y, w, b, m, n):
    for _ in range(n):
        dj_w, dj_b = derive(x, y, m, w, b)
        w -= a * dj_w
        b -= a * dj_b
    return w, b
    
#### VARIABLES ####
p = int(input("How many features: "))
s = int(input("How many data sets: "))

x_arr = np.empty((0, p))

for i in range(s):
    temp_row = list(map(int, input(f"Input data set {i+1} of X: ").split()))
    x_arr = np.vstack([x_arr, temp_row])
    
y_arr = np.array(list(map(int, input("Input Train Set Y: ").split())))

w_arr = np.zeros(p)
b = 0
m = len(x_arr)   # number of training examples
n = 10000
a = 0.01

#### TRAIN ####
w_arr, b = gradient_descent(a, x_arr, y_arr, w_arr, b, m, n)

b = round(b, 2)

for k in range(len(w_arr)):
    w_arr[k] = round(w_arr[k], 2)

print(f"y = {w_arr}.x_vec + {b}")
