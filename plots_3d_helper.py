# This file contains all plotting functions
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import numpy as np

# 3D plotting
def plotter3D(vars_3d, title):
    nx, ny, nz = vars_3d.shape
    #norm = matplotlib.colors.Normalize(vmin=np.min(vars_3d), vmax=np.max(vars_3d))
    norm = matplotlib.colors.Normalize(vmin=np.min(vars_3d), vmax=np.max(vars_3d))
    x, y, z = np.indices((nx + 1, ny + 1, nz + 1))
    
    plt.rc('font', size=12)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    # ax.set_box_aspect((1, 1, 1))
    #ax.invert_zaxis()
    ax.invert_yaxis()
    # ax.invert_xaxis()
    
    ax.set_axis_off()
    
    # Add axis direction labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    # Hide ticks and grid lines, but keep labels
    ax.grid(False)  # Disable grid lines
    
    ax.view_init(elev=30, azim=0)  # 60, -60
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1, 1]))
    # plotting black boundary box
    ax.plot3D([0, nx], [0, 0], [0, 0], 'k', zorder=1e6, linewidth=0.5)
    ax.plot3D([0, nx], [0, 0], [nz, nz], 'k', linewidth=0.5)
    ax.plot3D([0, nx], [ny, ny], [0, 0], 'k', zorder=1e6, linewidth=0.5)
    ax.plot3D([0, nx], [ny, ny], [nz, nz], 'k', zorder=1e6, linewidth=0.5)
    ax.plot3D([0, 0], [0, ny], [0, 0], 'k', zorder=1e6, linewidth=0.5)
    ax.plot3D([nx, nx], [0, ny], [0, 0], 'k', zorder=1e6, linewidth=0.5)
    ax.plot3D([nx, nx], [0, ny], [nz, nz], 'k', zorder=1e6, linewidth=0.5)
    ax.plot3D([0, 0], [0, ny], [nz, nz], 'k', zorder=1e6, linewidth=0.5)
    ax.plot3D([nx, nx], [ny, ny], [0, nz], 'k', zorder=1e6, linewidth=0.5)
    ax.plot3D([nx, nx], [0, 0], [0, nz], 'k', zorder=1e6, linewidth=0.5)
    ax.plot3D([0, 0], [0, 0], [0, nz], 'k', zorder=1e6, linewidth=0.5)
    ax.plot3D([0, 0], [ny, ny], [0, nz], 'k', zorder=1e6, linewidth=0.5)
    
    
    colors = plt.cm.jet(norm(vars_3d.transpose((0, 1, 2))))
    filled = np.where(vars_3d.transpose((0, 1, 2)) > 0, np.ones((nx, ny, nz)), np.zeros((nx, ny, nz)))
    
    vox = ax.voxels(x, y, z, filled, facecolors=colors,
                    edgecolors=colors,  # brighter
                    linewidth=0.5,
                    alpha=1, shade=False)
    
    
    m = cm.ScalarMappable(cmap=cm.jet, norm=norm)
    m.set_array([])
    plt.colorbar(m, ax=ax, shrink=0.5, pad=0.001)
    # plt.title(title)
    plt.tight_layout()
    plt.savefig(title+'.png', dpi=600)
    plt.show()


def plotter3D_sub(ax, vars_3d, title=""):
    nx, ny, nz = vars_3d.shape
    norm = matplotlib.colors.Normalize(vmin=np.min(vars_3d), vmax=np.max(vars_3d))

    x, y, z = np.indices((nx + 1, ny + 1, nz + 1))
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    ax.invert_yaxis()
    ax.set_axis_off()

    # boundary box
    ax.plot3D([0, nx], [0, 0], [0, 0], 'k',  linewidth=0.3)
    ax.plot3D([0, nx], [0, 0], [nz, nz], 'k', linewidth=0.3)
    ax.plot3D([0, nx], [ny, ny], [0, 0], 'k', zorder=1e6, linewidth=0.3)
    ax.plot3D([0, nx], [ny, ny], [nz, nz], 'k', zorder=1e6, linewidth=0.3)
    ax.plot3D([0, 0], [0, ny], [0, 0], 'k', zorder=1e6, linewidth=0.3)
    ax.plot3D([nx, nx], [0, ny], [0, 0], 'k', zorder=1e6, linewidth=0.3)
    ax.plot3D([nx, nx], [0, ny], [nz, nz], 'k', zorder=1e6, linewidth=0.3)
    ax.plot3D([0, 0], [0, ny], [nz, nz], 'k', zorder=1e6, linewidth=0.3)
    ax.plot3D([nx, nx], [ny, ny], [0, nz], 'k', zorder=1e6, linewidth=0.3)
    ax.plot3D([nx, nx], [0, 0], [0, nz], 'k', zorder=1e6, linewidth=0.3)
    ax.plot3D([0, 0], [0, 0], [0, nz], 'k', zorder=1e6, linewidth=0.3)
    ax.plot3D([0, 0], [ny, ny], [0, nz], 'k', zorder=1e6, linewidth=0.3)

    colors = plt.cm.jet(norm(vars_3d))
    filled = vars_3d > 0.0

    ax.voxels(
        x, y, z,
        filled,
        facecolors=colors,
        edgecolors=colors,
        linewidth=0.05,
        alpha=1.0,
        shade=False
    )

    ax.set_title(title, fontsize=10)


def compare_rollout_8panel(pred_seq, true_seq, timesteps, savefig=None):
    fig = plt.figure(figsize=(14, 7))

    for j, k in enumerate(timesteps):
        # predicted saturation
        ax = fig.add_subplot(2, 4, j+1, projection='3d')
        sat_pred = pred_seq[k, 1].copy()
        sat_pred[sat_pred < 0.05] = 0.0
        plotter3D_sub(ax, sat_pred, title=f"Pred t={5*(k+1)}y")

        # true saturation
        ax = fig.add_subplot(2, 4, j+5, projection='3d')
        sat_true = true_seq[k, 1].copy()
        sat_true[sat_true < 0.05] = 0.0
        plotter3D_sub(ax, sat_true, title=f"True t={5*(k+1)}y")

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=600)
    plt.show()


