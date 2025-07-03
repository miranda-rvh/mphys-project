import numpy as np
import matplotlib.pyplot as plt
import mplhep
mplhep.style.use("LHCb2")

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

data = np.genfromtxt("beta_vals_cxl.txt", dtype = 'float', skip_header=True, delimiter=',')
save_prefix = '/home/miran/mphys/SiC_Diode/plots/SiC/depth_scan_CX/'

x_coord = data[:,0].astype(int)

y_coord = data[:,1].astype(int)

beta = data[:,2]
beta_error = data[:,3]

unique_x = np.unique(x_coord)
unique_y = np.unique(y_coord)

grid = np.zeros((5,5))

print(np.median(beta), "+/-", np.std(beta))
print(np.min(beta), np.max(beta))

for i in range(len(x_coord)):
    x_id = np.where(unique_x==x_coord[i])[0][0]
    y_id = np.where(unique_y==y_coord[i])[0][0]

    grid[y_id, x_id] = beta[i]

plt.clf()
fig, ax = plt.subplots()
cax = ax.imshow(grid,origin="lower",cmap="bone", extent=[x_coord.min() - 0.5, x_coord.max() + 0.5, y_coord.min() - 0.5,y_coord.max() + 0.5], \
        vmin = 1.9e-14, vmax=2.3e-14)
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label(r"$\beta_2$ [m/W]")
ax.set_xlabel("x coordinate")
ax.set_ylabel("y coordinate")
ax.set_xticks(unique_x)
ax.set_yticks(unique_y)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
plt.savefig(save_prefix + "beta2_grid_cxl.png")
plt.clf()
