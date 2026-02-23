import numpy as np
import matplotlib.pyplot as plt

# Parameters matching the real distribution code
t = 1000
log_base = 10.0
start_exp = 0.0
log_scale = 0.001  # same as in more_smdp_envs.py

# Step-based scaling (like the real SinLogDist/CosLogDist)
steps = np.arange(1, t + 1)
log_multiplier1 = log_base ** (start_exp + steps * log_scale)
log_multiplier2 = log_base ** (start_exp + steps * log_scale / 4)

cos_line = (np.cos(steps) + 2) * log_multiplier2
sin_line = (np.sin(steps) + 15) * log_multiplier1

# Avoid division by zero
division = np.divide(cos_line, sin_line)  # , out=np.zeros_like(cos_line), where=sin_line != 0)

plt.figure(figsize=(12, 8))

# First subplot: sin and cos with log scaling
plt.subplot(1, 1, 1)
plt.plot(cos_line, label="cos * log_scale", color="blue")
plt.plot(sin_line, label="sin * log_scale", color="orange")
plt.plot(division, label="cos / sin", color="red", alpha=0.7)
plt.yscale("symlog")
plt.xlabel("Step")
plt.ylabel("Value")
plt.title(f"Sin Log Scaling / Cos Log Scaling (log_base={log_base}, log_scale={log_scale})")
plt.legend()
plt.grid()

# Second subplot: cumsum of the division
# plt.subplot(2, 1, 2)
# cumsum_division = np.cumsum(division)
# plt.plot(cumsum_division, label="cumsum(cos / sin)", color="purple")
#
# # Add cumsum of a linear slope line for comparison
# linear_line = np.arange(1, t + 1)  # linear slope: 1, 2, 3, ...
# cumsum_linear = np.cumsum(linear_line/20)
# plt.plot(cumsum_linear, label="cumsum(linear slope)", color="green", linestyle="--")
#
# plt.xlabel("Step")
# plt.ylabel("Cumulative Sum")
# plt.title("Cumulative Sum of cos/sin Division vs Linear")
# plt.legend()
# plt.grid()

plt.tight_layout()
plt.show()
