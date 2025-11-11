import json
import matplotlib.pyplot as plt
import numpy as np
import math
import os

# Ensure output directory exists
os.makedirs("bpp_histograms", exist_ok=True)

# Load JSON
with open("hard_parsed.json", "r") as f:
    data = json.load(f)

# --- Histogram 1: items per instance ---
instance_names = list(data.keys())
num_items = [len(info["items"]) for info in data.values()]

plt.figure(figsize=(10, 5))
plt.bar(instance_names, num_items)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of items")
plt.xlabel("Instance name")
plt.title("Number of items per instance")
plt.tight_layout()
plt.savefig("bpp_histograms/num_of_items.png")
plt.close()

# --- Combined Histogram 2: all instances together ---
bins = np.arange(0, 1100, 100)
bin_labels = [f"{b}-{b+100}" for b in bins[:-1]]

n_instances = len(data)
cols = 3  # adjust layout if needed
rows = math.ceil(n_instances / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
axes = axes.flatten()

for i, (instance, info) in enumerate(data.items()):
    items = np.array(info["items"])
    counts, _ = np.histogram(items, bins=bins)

    ax = axes[i]
    ax.bar(bin_labels, counts)
    ax.set_title(instance)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

# Hide unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Item Size Distribution per Instance", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("bpp_histograms/all_item_size_histograms.png", dpi=300)
plt.close()

