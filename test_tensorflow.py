import matplotlib.pyplot as plt

# ax = 1
# ay = 1
# az = 1
# plt.sphere(20)
# plt.sphere(ax, ay, az)


[x, y, z] = plt.sphere(5)
plt.surface(x, y, z)
plt.xlabel('x')
plt.view(3)

