import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mplhep
mplhep.style.use("LHCb2")

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

save_prefix = '/home/miran/mphys/SiC_Diode/plots/SiC/depth_scan_TCT/'
data = np.genfromtxt("beta_vals_tct.txt", delimiter=",", dtype=[('coordinate', 'U10'), ('beta', 'f8'), ('error', 'f8')], skip_header=True)

coordinates = data['coordinate']

x_coords = np.array([int((coord[1]) )for coord in coordinates])

y_coords = np.array([int((coord[3])) for coord in coordinates])


beta_values = data['beta']

unique_x = np.unique(x_coords)
unique_y = np.unique(y_coords)

grid = np.zeros((5,5))

for i in range(len(x_coords)):
    x_id = np.where(unique_x==x_coords[i])[0][0]
    y_id = np.where(unique_y==y_coords[i])[0][0]

    grid[y_id, x_id] = beta_values[i]
print(np.median(beta_values), "+/-", np.std(beta_values))
print(np.min(beta_values), np.max(beta_values))


plt.clf()
fig, ax = plt.subplots()
cax = ax.imshow(grid, origin="lower", cmap="bone", extent=[x_coords.min() - 0.5, x_coords.max() + 0.5, \
        y_coords.min() - 0.5,y_coords.max() + 0.5], vmin=5.5e-14, vmax=8e-14)
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label(r"$\beta_2$ [m/W]")
x5, y5 = 5, 5
rect = Rectangle(
    (x5 - 0.5, y5 - 0.5),
    width=1,
    height=1,
    linewidth=0,
    edgecolor='black',
    facecolor='white',
    hatch='///',  # Diagonal lines
    alpha=0.5
)
ax.add_patch(rect)
ax.set_xlabel("x coordinate")
ax.set_ylabel("y coordinate")
ax.set_xticks(unique_x)
ax.set_yticks(unique_y)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
plt.savefig(save_prefix + "beta2_grid_tct.png")
plt.clf()