import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# -----------------------------
# Global matplotlib settings
# -----------------------------
plt.rcParams.update({
    "font.size": 25,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "pdf.fonttype": 42,   # embed fonts in PDF
    "ps.fonttype": 42,
})

# -----------------------------
# Data
# -----------------------------fax.
horizon = np.array([10, 25, 50, 75, 100, 200, 500, 1000, 2000, 3000])

data = {
    "GPU-SLS (Nominal)":      [2.854, 3.36, 3.64, 4.94, 6.06, 8.5, 11.1, 17.2, 26.78, 38.1],
    "ADMM Float64":           [4.11, 5.03, 5.97, 7.72, 10.23, 14.99, 29.07, 51.07, 93.3, 140.5],
    "acados (HPIPM)":         [0.773, 1.46, 2.6, 5.19, 6.72, 12.08, 38.02, 127.8, 583, 1716],
    "acados (OSQP)":          [3.08, 4.57, 9.88, 13.31, 18.96, 45.6, 305, 1777, 12474, np.nan],
    "primal-dual iLQR AL":    [5.26, 6.14, 8.75, 11.326, 15.49, 20.1, 33.26, 60.9, np.nan, np.nan],
    "cparcon-IPM":            [692, 1835, 8675, 159901, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    "cparcon-ADMM":           [1107, 3530, 14189, 61521, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
}

# -----------------------------
# Figure size (in inches)
# -----------------------------
# Recommended for full-width / two-column figure
fig_width = 18   # inches
fig_height = 10  # inches

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# -----------------------------
# Plot
# -----------------------------
for label, values in data.items():
    y = np.array(values, dtype=float)
    mask = ~np.isnan(y)
    ax.plot(
        horizon[mask],
        y[mask],
        marker="o",
        linewidth=4,
        markersize=8,
        label=label,
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Horizon Length", fontsize=30)
ax.set_ylabel("Solve Time (ms)", fontsize=30)
# ax.set_title("Solver Scaling vs Horizon Length (Log Scale)")
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

# Extend x-range to create space for legend INSIDE axes
ax.set_xlim(horizon.min(), horizon.max() * 1.6)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.58, 1.02),  # centered below axes
    ncol=1,                       # adjust columns as needed
    frameon=False,
    fontsize=30,
    columnspacing=1.,
    handlelength=1.5,
    labelspacing=0.2,
)

ax.plot(
    1000,
    60.9,
    marker="x",
    markersize=18,
    markeredgewidth=4,
    linestyle="None",
    color="black",   # remove if you want default color cycle
    zorder=10,
)

# Leave space at the bottom for the legend
fig.tight_layout(rect=[0, 0.12, 1, 1])



# Reserve space on the right for the legend
fig.tight_layout(rect=[0, 0, 0.72, 1])

# -----------------------------
# Save / Show
# -----------------------------
fig.savefig("solver_scaling_log.pdf", bbox_inches="tight")
plt.show()
print('done')
