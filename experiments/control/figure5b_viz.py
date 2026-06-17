import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Load data (shape: 14 runs × 100 episodes) ──────────────────────────────
data = {
    "QRC":        np.array([np.load("CartPole/"+str(i)+"dataQRC.npy") for i in [0,15,29]]).reshape(42,100),
    "QC":         np.array([np.load("CartPole/"+str(i)+"dataQC.npy") for i in [0,15,29]]).reshape(42,100),
    "Q-Learning": np.array([np.load("CartPole/"+str(i)+"dataQLearning.npy") for i in [0,15,29]]).reshape(42,100),
}
print(data)
colors = {
    "QRC":        "#2563EB",
    "QC":         "#16A34A",
    "Q-Learning": "#DC2626",
}

# ── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))

for label, arr in data.items():
    print(arr.shape)
    episodes = np.arange(1, arr.shape[1] + 1)
    mean  = arr.mean(axis=0)
    std   = arr.std(axis=0)
    color = colors[label]

    ax.plot(episodes, mean, label=label, color=color, linewidth=2)
    #ax.fill_between(episodes, mean - std, mean + std,
                    #color=color, alpha=0.15)

# ── Formatting ──────────────────────────────────────────────────────────────
ax.set_xlabel("Episode", fontsize=13)
ax.set_ylabel("Steps per Episode", fontsize=13)
ax.set_title("Steps per Episode — QRC vs QC vs Q-Learning", fontsize=14, fontweight="bold")
ax.set_xlim(1, 100)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.7)
ax.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.4)
ax.legend(fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.savefig("cartpole.png", dpi=150, bbox_inches="tight")
print("Saved.")
