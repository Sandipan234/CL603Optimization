import numpy as np
import matplotlib.pyplot as plt

x1 = np.arange(-1, 1, (1 / 30))
x2 = np.arange(-1, 1, (1 / 30))

# line plot

y_pos = (1 - x1) ** 4 + (8 * x1) - x1 + 4
y_neg = (1 + x1) ** 4 - (8 * x1) - x1 + 2

plt.figure(1)
plt.plot(x1, y_pos, label='x2= +1', linestyle='dashed')
plt.plot(x1, y_neg, label='x2= -1', linestyle='solid')
plt.title("Test function versus x1 for a fixed x2")
plt.legend()
plt.grid()
plt.show(block=False)

# Surface Plot

plt.figure(2)
X1, X2 = np.meshgrid(x1, x2)
y = (X2 - X1) ** 4 + (8 * X1 * X2) - X1 + X2 + 3
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, y)
plt.title("Surface Plot of test function")
plt.show(block=False)

# Contour Plot
fig, axl = plt.subplots(figsize=(10, 6))
cs = axl.contour(X1, X2, y)
plt.clabel(cs)
axl.set_title("contour plot of test function")
axl.set_xlabel("x1")
axl.set_ylabel("x2")
plt.show()
