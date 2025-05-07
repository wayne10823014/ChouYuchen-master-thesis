import matplotlib.pyplot as plt
import numpy as np

# GPU names
gpus = ['RTX 2070 S', 'Apple M1 GPU', 'Intel UHD 620']

# Predicted and measured percentages (relative to RTX 2070 S CUDA)
predicted = [119, 417, 3097]
measured  = [119, 262, 1762]

x = np.arange(len(gpus))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, predicted, width, label='Predicted (TFLOPS‑scaled)')
bars2 = ax.bar(x + width/2, measured,  width, label='Measured (WebGPU)')

# Annotate bars
def attach_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom')

attach_labels(bars1)
attach_labels(bars2)

ax.set_xticks(x)
ax.set_xticklabels(gpus)
ax.set_ylabel('Execution time relative to RTX 2070 S CUDA (%)')
ax.set_title('WebGPU Optimized: Predicted vs. Measured (N = 100 000)')
ax.legend()
plt.tight_layout()
plt.show()
