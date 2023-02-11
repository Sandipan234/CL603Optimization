import numpy as np

# Defining the Input vector and also checking for invalid input from the user
x_input = []
no_input_error = True

# taking input from the user
try:
    ele1 = float(input('enter the 1st element \n'))
    x_input.append(ele1)
    ele2 = float(input('enter the 2nd element \n'))
    x_input.append(ele2)
    x_input = np.array(x_input)
    print(x_input)
except ValueError:
    print("Error in input")
    no_input_error = False

if no_input_error:
    # Defining the Function
    def math_function(z):
        x1, x2 = z
        fx = 2 * np.exp(x1) * x2 + 3 * x1 * (x2 ** 2)
        return fx


    # print(math_function(x_input))

    # differential element
    h = 0.001
    del_x = 0.001

    # Function to calculate Gradient
    def compute_gradient(func, b):
        gradient = np.array([])  # Array to store the value of gradient
        # Looping through the input Vector to find each term of the gradient
        for i in range(len(b)):
            e = np.zeros(len(b))  # increment vector
            e[i] = 1  # setting the ith row to 1 for perturbation of the ith term
            gradient_value = (func(b + h * e) - func(b - h * e)) / (2 * h)  # calculating the ith partial derivative
            gradient = np.append(gradient, gradient_value)
            gradient = np.array([gradient]).T
        return gradient


    # Printing the Gradient
    print(compute_gradient(math_function, x_input))

    # Function to calculate the Hessian
    def compute_hessian(func, b):
        hessian = np.empty((len(b), len(b)), float)  # initialising the hessian matrix
        for i in range(len(b)):
            for j in range(len(b)):
                ei = np.zeros(len(b))  # arrays to help calculate the value at the desired index for xi
                ej = np.zeros(len(b))  # arrays to help calculate the value at the desired index for xj
                if i == j:
                    ei[i] = 1  # setting value of xi =1 for perturbation of i,j th term
                    calc = (func(b + del_x * ei) - 2 * func(b) + func(b - del_x * ei)) / (
                            del_x ** 2)  # calculating the diagonal value
                    hessian[i][j] = calc
                else:
                    ei[i] = 1  # setting value of xi =1 for perturbation of i,j th term
                    ej[j] = 1  # setting value of xj =1 for perturbation of i,j th term
                    calc = (func(b + del_x * ei + del_x * ej) + func(b - del_x * ei - del_x * ej) - func(
                        b - del_x * ei + del_x * ej) - func(b + del_x * ei - del_x * ej)) / (
                                   4 * del_x * del_x)  # calculating the off-diagonal value
                    hessian[i][j] = calc
        hessian = np.array([hessian]).T
        return hessian


    # Printing the Hessian
    print(compute_hessian(math_function, x_input))
