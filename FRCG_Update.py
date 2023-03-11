"""=================================================== Assignment 4 ===================================================

Some instructions:
    * You can write seperate function for gradient and hessian computations.
    * You can also write any extra function as per need.
    * Use in-build functions for the computation of inverse, norm, etc.

"""

""" Import the required libraries"""

# Start your code here

import numpy as np


# End your code here


def func(x_input):
    """
    --------------------------------------------------------
    Write your logic to evaluate the function value.

    Input parameters:
        x: input column vector (a numpy array of n dimension)

    Returns:
        y : Value of the function given in the problem at x.

    --------------------------------------------------------
    """

    # Start your code here
    x1, x2 = x_input
    # y = -0.0001 * (abs(np.sin(x1) * np.sin(x2) * np.exp(abs(100 - np.sqrt(x1 * 2 + x2 * 2) / np.pi)))) ** 0.1
    y = (x1 - 1) ** 2 + (x2 - 1) ** 2 - x1 * x2
    # y = x1**2+x2**2+(0.5*x1+x2)**2+(0.5*x1+x2)**4
    # y = (x1+2*x2-7)**2+(2*x1+x2-5)**2
    # End your code here

    return y


def gradient(func, x_input):
    h = 0.001
    grad_f = np.array([])
    for i in range(len(x_input)):
        e = np.array([np.zeros(len(x_input), dtype=int)]).T
        e[i][0] = 1
        del_f = (func(x_input + (h * e)) - func(x_input - (h * e))) / (2 * h)
        grad_f = np.append(grad_f, del_f)
    delF = np.array([grad_f]).T
    return delF


def backtracking_line_search(f, x, direction, alpha=5, beta=0.8):
    # Evaluate function and gradient at current point
    fx = f(x)
    gx = gradient(f, x).T
    while f(x + alpha * direction) > fx + 0.1 * alpha * beta * gx.dot(direction):
        alpha *= beta

    return alpha


def FRCG(func, x_initial):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for FR-CG using in-exact line search.

    Input parameters:
        func : input function to be evaluated
        x_initial: initial value of x, a column vector (numpy array)

    Returns:
        x_output : converged x value, a column vector (numpy array)
        f_output : value of f at x_output
        grad_output : value of gradient at x_output, a column vector(numpy array)
    -----------------------------------------------------------------------------------------------------------------------------
    """

    # Start your code here
    eps = 1e-6
    max_iter = 15000
    xk = x_initial
    p = -gradient(func, xk)
    for i in range(max_iter):
        alpha = backtracking_line_search(func, xk, p)
        grad0 = gradient(func, xk)
        xk = xk + alpha * p
        grad1 = gradient(func, xk)
        if np.linalg.norm(gradient(func, xk)) ** 2 < eps:
            break
        beta = np.dot(grad1.T, grad1) / np.dot(grad0.T, grad0)
        p = -grad1 + beta * p

    x_output = xk
    f_output = func(xk)
    grad_output = gradient(func, xk)
    # End your code here

    return x_output, f_output, grad_output


"""--------------- Main code: Below code is used to test the correctness of your code ---------------

    func : function to evaluate the function value. 
    x_initial: initial value of x, a column vector, numpy array

"""

x_initial = np.array([[1.5, 1.5]]).T

x_output, f_output, grad_output = FRCG(func, x_initial)

# print(x_output)
# print('\n next \n')
# print(f_output)
# print('\n next \n')
# print(grad_output)
