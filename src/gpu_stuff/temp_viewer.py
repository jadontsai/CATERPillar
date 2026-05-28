import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#quick and dirty 3D plotting
# csv version
df = pd.read_csv("final.csv")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for object_id, g in df.groupby("object_id"):
    g = g.sort_values("sphere_id")
    ax.plot(g["x"], g["y"], g["z"], linewidth=1)
    ax.scatter(g["x"], g["y"], g["z"], s=(g["r"]*10) ** 2)

#swc version
# df = pd.read_csv("glial_test_2.csv", sep=r"\s+")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# for object_id, g in df.groupby("component_id"):
#     #g = g.sort_values("sphere_id") #should be linear already... (like increasing)
#     ax.plot(g["X"], g["Y"], g["Z"], linewidth=1)
#     ax.scatter(g["X"], g["Y"], g["Z"], s=(g["outer_radius"] * 20) ** 2)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()

#trying to plot the swc output, not sure if this is the way to go
# df = pd.read_csv("glial_test_2.csv", sep=r"\s+")

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(projection="3d")

# # sphere mesh
# u = np.linspace(0, 2*np.pi, 16)
# v = np.linspace(0, np.pi, 8)
# unit_x = np.outer(np.cos(u), np.sin(v))
# unit_y = np.outer(np.sin(u), np.sin(v))
# unit_z = np.outer(np.ones_like(u), np.cos(v))

# colors = plt.cm.tab20(np.linspace(0, 1, df["cell_id"].nunique()))

# for color, (cell_id, g) in zip(colors, df.groupby("cell_id")):
#     for _, row in g.iterrows():
#         r = row["outer_radius"]
#         x = row["X"] + r * unit_x
#         y = row["Y"] + r * unit_y
#         z = row["Z"] + r * unit_z

#         ax.plot_surface(
#             x, y, z,
#             color=color,
#             alpha=0.35,
#             linewidth=0,
#             shade=True
#         )

#     ax.plot(g["X"], g["Y"], g["Z"], color=color, linewidth=1.5, label=f"axon {cell_id}")

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_box_aspect((
#     df["X"].max() - df["X"].min(),
#     df["Y"].max() - df["Y"].min(),
#     df["Z"].max() - df["Z"].min()
# ))

# ax.legend()
# plt.show()