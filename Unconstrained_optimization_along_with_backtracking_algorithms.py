"""=================================================== Assignment 4 ===================================================

Some instructions:
    * You can write seperate function for gradient and hessian computations.
    * You can also write any extra function as per need.
    * Use in-build functions for the computation of inverse, norm, etc.

"""

""" Import the required libraries"""


# Start your code here
import numpy as np
import matplotlib.pyplot as plt


# End your code here


def func(x_input):


    # Start your code here
    x1, x2 = x_input
    y = (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2
    # End your code here

    return y




def gradient(func, x_input):
    h = 0.001
    grad_f = np.array([])
    for i in range(len(x_input)):
        e = np.array([np.zeros(len(x_input), dtype=int)]).T
        e[i][0] = 1
        del_f = (func(x_input + (h*e)) - func(x_input - (h*e)))/ (2*h)
        grad_f = np.append(grad_f, del_f)
    delF = np.array([grad_f]).T
    return delF




def hessian(func, x_input):

    # Start your code here
    # Use the code from assignment 2
    n = len(x_input)
    del_x = np.full(shape=n, fill_value=0.001)
    del2F = np.array([]).reshape(0, n)
    for i in range(n):
        hess_f = np.array([])
        del_i = np.array([np.zeros(n)]).T
        del_i[i][0] = del_x[i]
        for j in range(n):
            del_j = np.array([np.zeros(n)]).T
            del_j[j][0] = del_x[j]
            if (i == j):
                a = x_input + del_i
                b = x_input - del_j
                value = (func(a) - (2 * func(x_input)) + func(b)) / (del_x[i] ** 2)
                hess_f = np.append(hess_f, value)
            else:
                a = x_input + del_i + del_j
                b = x_input - del_i - del_j
                c = x_input - del_i + del_j
                d = x_input + del_i - del_j
                value = (func(a) + func(b) - func(c) - func(d)) / (4 * del_x[i] * del_x[j])
                hess_f = np.append(hess_f, value)
        del2F = np.vstack([del2F, hess_f])
    return del2F

    # End your code here


def backtracking_line_search(f, x, direction, alpha=5, beta=0.8, tol=1e-6):
    # Evaluate function and gradient at current point
    fx = f(x)
    gx = gradient(f, x).T

    while f(x + alpha * direction) > fx + 0.1 * alpha * beta * gx.dot(direction):
        alpha *= beta

    return alpha


def steepest_descent(func, x_initial):


    # Start your code here

    # Start your code here
    N = 15000
    eps = 10 ** (-6)
    x_iters = []
    f_iters = []
    x_iters.append(x_initial.T[0])
    f_iters.append(func(x_initial))
    x = x_initial
    i = 0
    while i <= N:
        alpha = backtracking_line_search(func, x, -gradient(func, x))
        x = x - alpha * gradient(func, x)
        i += 1
        x_iters.append(x.T[0])
        f_iters.append(func(x))
        if np.linalg.norm(gradient(func, x)) ** 2 < eps:
            break
        else:
            continue
    x_output = x
    f_output = func(x)
    grad_output = gradient(func, x)

    q = np.zeros(len(x_iters))
    w = np.zeros(len(x_iters))
    z = np.ones(i + 1)
    for j in range(0, i):
        q[j] = x_iters[j][0]
        w[j] = x_iters[j][1]
        z[j] = z[j] + i
    plt.figure()
    plt.plot(z, q, label='x1', linestyle='dashed')
    plt.plot(z, w, label='x2', linestyle='solid')
    plt.title("iterations versus x1,x2")
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(z, f_iters, label='x1', linestyle='dashed')
    plt.show()
    # End your code here

    return x_output, f_output, grad_output


def newton_method(func, x_initial):

    # Start your code here
    N = 15000
    eps = 10 ** (-6)
    x = x_initial
    i = 0
    while i <= N:
        alpha = backtracking_line_search(func, x, -gradient(func, x))
        grad = -gradient(func, x)
        hess = np.linalg.inv(hessian(func, x))
        x = x + alpha * (np.matmul(hess, grad))
        if np.linalg.norm(gradient(func, x)) ** 2 < eps:
            break
        else:
            continue
    x_output = x
    f_output = func(x)
    grad_output = gradient(func, x)
    # End your code here

    return x_output, f_output, grad_output


def bfgs_update(x, s, y, Hk):
    y1 = np.squeeze(np.asarray(y))
    s1 = np.squeeze(np.asarray(s))
    rho = 1 / np.dot(y1, s1)

    I = np.identity(len(x))
    u = s - np.dot(Hk, y)
    v = np.outer(u, u)

    Hk_new = (I - rho * np.outer(s, y)) @ Hk @ (I - rho * v) + rho * np.outer(s, s)

    return Hk_new

def quasi_newton_method(func, x_initial):


    # Start your code here
    c = np.identity(len(x_initial), dtype=float)
    N = 15000
    eps = 10 ** (-6)
    x = x_initial
    i = 0
    s = x
    y = x
    while i <= N:
        alpha = backtracking_line_search(func, x, -gradient(func, x))
        grad = gradient(func, x)
        x1 = x
        grad1 = gradient(func, x1)
        x = x - alpha * np.matmul(c, grad)
        x2 = x
        grad2 = gradient(func, x2)
        s = x2 - x1
        y = grad2 - grad1
        c = bfgs_update(x, s, y, c)
        if np.linalg.norm(gradient(func, x)) ** 2 < eps:
            break
        else:
            continue
    x_output = x
    f_output = func(x)
    grad_output = gradient(func, x)
    # End your code here

    return x_output, f_output, grad_output


def iterative_methods(func, x_initial):

    x_SD, f_SD, grad_SD = steepest_descent(func, x_initial)
    x_NM, f_NM, grad_NM = newton_method(func, x_initial)
    x_QN, f_QN, grad_QN = quasi_newton_method(func, x_initial)

    return x_SD, f_SD, grad_SD, x_NM, f_NM, grad_NM, x_QN, f_QN, grad_QN


# Define x_initial here

x_initial=np.array([[1.5,1.5]]).T
x_SD, f_SD, grad_SD, x_NM, f_NM, grad_NM, x_QN, f_QN, grad_QN = iterative_methods(func, x_initial)
