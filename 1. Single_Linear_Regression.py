#### FUNCTIONS ####

'''
def predict(x, m, w, b):
    tmp_arr = []

    for i in range(m):
        tmp_arr.append(w * x + b)

    return tmp_arr


def cost(y_tru, y_pre, m):

    sum_cost = 0

    for i in range(m):
        j = (y_tru[i] - y_pre[i]) ** 2
        sum_cost += j

    cost = sum_cost / m
    
    return cost
'''

def derive(x, y, m, w, b, b_w):

    dj_w, dj_b = 0, 0

    for i in range(m):
        dj_b_i = (w * x[i] + b - y[i])
        dj_w_i = dj_b_i * x[i]
        dj_b += dj_b_i
        dj_w += dj_w_i

    dj_b /= m
    dj_w /= m

    if (b_w == 'b'):
        return dj_b
    elif (b_w == 'w'):
        return dj_w

def gradient_descent(a, x, y, w, b, m, n):
    
    for i in range(n):
        w_tmp = w - a * derive(x, y, m, w, b, 'w')
        b_tmp = b - a * derive(x, y, m, w, b, 'b')
        w = w_tmp
        b = b_tmp

    return w, b
    
#### VARIABLES ####

x_arr = list(map(int, input("Input Train Set X: ").split()))
y_arr = list(map(int, input("Input Train Set Y: ").split()))

w, b = 1, 0
m = len(x_arr)
n = 10000
a = 0.01

#### CODE ####

'''
y_pre = predict(x_arr, m, w, b)

cost = cost(y_arr, y_pre, m)
'''

w, b = gradient_descent(a, x_arr, y_arr, w, b, m, n)

w, b = round(w, 2), round(b, 2)

print(f"y = {w} * x + {b}")
