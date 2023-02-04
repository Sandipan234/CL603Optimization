"""============================================ Assignment 3: Newton Method ============================================"""

""" Import the required libraries"""
import numpy as np
import matplotlib.pyplot as plt


# Start you code here


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
    x1, x2 = x_input[0][0], x_input[1][0]
    y = x1 ** 2 + x2 ** 2 + (0.5 * x1 + x2) ** 2 + (0.5 * x1 + x2) ** 4
    # End your code here

    return y


def gradient(func, x_input):
    """
    --------------------------------------------------------------------------------------------------
    Write your logic for gradient computation in this function. Use the code from assignment 2.

    Input parameters:
      func : function to be evaluated
      x_input: input column vector (numpy array of n dimension)

    Returns:
      delF : gradient as a column vector (numpy array)
    --------------------------------------------------------------------------------------------------
    """
    # Start your code here
    # Use the code from assignment 2
    h = 0.001
    grad_f = np.array([])
    for i in range(len(x_input)):
        e = np.array([np.zeros(len(x_input), dtype=int)]).T
        e[i][0] = 1
        del_f = (func(x_input + (h * e)) - func(x_input - (h * e))) / (2 * h)
        grad_f = np.append(grad_f, del_f)
        delF = np.array([grad_f]).T
    # End your code here
    try:
        return delF
    except:
        z = "Invalid Input"
        return z


def hessian(func, x_input):
    """
    --------------------------------------------------------------------------------------------------
    Write your logic for hessian computation in this function. Use the code from assignment 2.

    Input parameters:
      func : function to be evaluated
      x_input: input column vector (numpy array)

    Returns:
      del2F : hessian as a 2-D numpy array
    --------------------------------------------------------------------------------------------------
    """
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
    # End your code here

    return del2F


def newton_method(func, x_initial):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for newton method in this function.

    Input parameters:
      func : input function to be evaluated
      x_initial: initial value of x, a column vector (numpy array)

    Returns:
      x_output : converged x value, a column vector (numpy array)
      f_output : value of f at x_output
      grad_output : value of gradient at x_output
      num_iterations : no. of iterations taken to converge (integer)
      x_iterations : values of x at each iterations, a (num_interations x n) numpy array where, n is the dimension of x_input
      f_values : function values at each iteration (numpy array of size (num_iterations x 1))
    -----------------------------------------------------------------------------------------------------------------------------
    """
    # Write code here
    eps = 10e-6
    N = 15000
    num_iterations = 0
    xk = x_initial

    x = [xk[0][0]]
    y = [xk[1][0]]
    f_values = []
    #    while (iterations<=15000)  (np.linalg.norm(func(xk)))**2>eps:
    while ((num_iterations <= 15000) and (np.linalg.norm(gradient(func, xk))) ** 2 > eps):
        grad = gradient(func, xk)
        hess = hessian(func, xk)
        q = np.linalg.inv(hess)
        xk = xk - np.matmul(q, grad)
        num_iterations += 1
        f_values.append(func(xk))
        x.append(xk[0][0])
        y.append(xk[1][0])
    f_output = func(xk)
    grad_output = gradient(func, xk)
    x_ar = np.array(x)
    y_ar = np.array(y)
    x_iterations = np.zeros((num_iterations, 2))
    x_output = xk

    for i in range(0, len(x) - 1):
        x_iterations[i][0] = x[i]
        x_iterations[i][1] = y[i]
    # End your code here

    return x_output, f_output, grad_output, num_iterations, x_iterations, f_values


def plot_x_iterations(NM_iter, NM_x):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for plotting x_input versus iteration number i.e,
    x1 with iteration number and x2 with iteration number in same figure but as separate subplots.

    Input parameters:
      NM_iter : no. of iterations taken to converge (integer)
      NM_x: values of x at each iterations, a (num_interations X n) numpy array where, n is the dimension of x_input

    Output the plot.
    -----------------------------------------------------------------------------------------------------------------------------
    """
    # Start your code here
    x1 = np.zeros(len(NM_x))
    x2 = np.zeros(len(NM_x))
    z = np.ones(NM_iter)
    for i in range(0, len(NM_x)):
        x1[i] = NM_x[i][0]
        x2[i] = NM_x[i][1]
        z[i] = z[i] + i
    plt.figure()
    plt.plot(z, x1, label='x1', linestyle='dashed')
    plt.plot(z, x2, label='x2', linestyle='solid')
    plt.xlabel("Number of iterations")
    plt.ylabel("X value")
    plt.legend()
    plt.show()
    # End your code here


def plot_func_iterations(NM_iter, NM_f):
    """
    ------------------------------------------------------------------------------------------------
    Write your logic to generate a plot which shows the value of f(x) versus iteration number.

    Input parameters:
      NM_iter : no. of iterations taken to converge (integer)
      NM_f: function values at each iteration (numpy array of size (num_iterations x 1))

    Output the plot.
    -------------------------------------------------------------------------------------------------
    """
    # Start your code here
    x = np.zeros(len(NM_f))
    z = np.ones(NM_iter)
    for i in range(0, NM_iter):
        z[i] = z[i] + i
    print(z)
    plt.figure()
    plt.plot(z, NM_f)
    plt.xlabel("Number of iterations")
    plt.ylabel("f_value")
    plt.show()
    # End your code here


"""--------------- Main code: Below code is used to test the correctness of your code ---------------"""

x_initial = np.array([[1.5, 1.5]]).T

x_output, f_output, grad_output, num_iterations, x_iterations, f_values = newton_method(func, x_initial)

print("\nFunction converged at x = \n", x_output)
print("\nFunction value at converged point = \n", f_output)
print("\nGradient value at converged point = \n", grad_output)

plot_x_iterations(num_iterations, x_iterations)

plot_func_iterations(num_iterations, f_values)
