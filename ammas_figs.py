import os

import numpy as np
import matplotlib.pyplot as plt

os.makedirs("AMMAS_figs", exist_ok=True)

# Parameters matching the real distribution code
t = 1000
log_base = 10.0
start_exp = 0.0
log_scale = 0.001  # same as in more_smdp_envs.py

steps = np.arange(1, t + 1)
log_multiplier1 = log_base ** (start_exp + steps * log_scale)
log_multiplier2 = log_base ** (start_exp + steps * log_scale / 2)

sin_line = (np.sin(steps) + 10) * log_multiplier1
cos_line = (np.sin(steps) + 10) * log_multiplier2

fig, ax = plt.subplots(figsize=(6, 6))

# Plot all lines on the same figure
ax.plot(sin_line, label=f"SinLogD (offset=10, log_scale={log_scale})", color="orange", linewidth=3)
ax.plot(cos_line, label=f"CosLogD (offset=10, log_scale={log_scale/2})", color="blue", linewidth=3)

division = np.divide(sin_line, cos_line)
ax.plot(division, label="SinLogD / CosLogD", color="red", linewidth=3)

ax.set_xlabel("Step", fontsize=14, fontweight='bold')
ax.set_ylabel("Value", fontsize=14, fontweight='bold')
ax.set_title("Drifting Sin Log, Cos Log, and Ratio", fontsize=16, fontweight='bold')
ax.tick_params(axis='both', labelsize=16, width=2, length=6)
ax.legend(fontsize=12, frameon=True, shadow=True)
ax.grid()

plt.tight_layout()
plt.savefig("AMMAS_figs/drifting_plots.png")
plt.show()
