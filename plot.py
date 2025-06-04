import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 500)
y = np.linspace(-4, 4, 500)
X, Y = np.meshgrid(x, y)
Z = 3*(X**2 + Y**2)**2 - 100*(X**2 - Y**2)

plt.contour(X, Y, Z, levels=[0], colors='blue')  # Leminiscate
plt.scatter([0, 10/np.sqrt(3), -10/np.sqrt(3)], [0, 0, 0], color='red', label='Roots')  # Roots
plt.xlabel('x'); plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5); plt.axvline(0, color='black', linewidth=0.5)
plt.grid(); plt.legend(); plt.title('Lemniscate with Roots'); plt.show()