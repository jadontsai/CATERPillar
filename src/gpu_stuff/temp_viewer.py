import pandas as pd
import matplotlib.pyplot as plt
#quick and dirty 3D plotting
# csv version
# df = pd.read_csv("final.csv")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# for object_id, g in df.groupby("object_id"):
#     g = g.sort_values("sphere_id")
#     ax.plot(g["x"], g["y"], g["z"], linewidth=1)
#     ax.scatter(g["x"], g["y"], g["z"], s=(g["r"] * 20) ** 2)

#swc version
df = pd.read_csv("final.csv", sep=r"\s+")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for object_id, g in df.groupby("object_id"):
    g = g.sort_values("sphere_id")
    ax.plot(g["x"], g["y"], g["z"], linewidth=1)
    ax.scatter(g["x"], g["y"], g["z"], s=(g["r"] * 20) ** 2)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()