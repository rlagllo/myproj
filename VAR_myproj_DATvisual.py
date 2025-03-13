import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("C:\\Users\\김해창\\Desktop\\airfoil data\\a18.dat", skiprows=1)

x = data[:,0]
y = data[:,1]

plt.plot(x,y)
plt.xlabel("X-axis label")
plt.ylabel("Y-axis label")
plt.xlim(0,1)  
plt.ylim(-0.3, 0.3)
plt.show()