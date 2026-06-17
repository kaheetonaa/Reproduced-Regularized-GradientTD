import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Load data (shape: runs × episodes) ──────────────────────────────────────
paths = {
    "QRC":       "CartPole/0dataQRC.npy",
    "QC":        "CartPole/0dataQC.npy",
    "Q-Learning":"CartPole/0dataQLearning.npy",
}

colors = {
    "QRC":        "#2563EB",   # blue
    "QC":         "#16A34A",   # green
    "Q-Learning": "#DC2626",   # red
}

data = {k: np.load(v) for k, v in paths.items()}

# ── Cumulative reward per run, then aggregate ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

for label, arr in data.items():
    cum = np.cumsum(arr, axis=1)          # (runs, episodes)
    mean = cum.mean(axis=0)
    std  = cum.std(axis=0)
    episodes = np.arange(1, mean.shape[0] + 1)

    c = colors[label]
    ax.plot(episodes, mean, label=label, color=c, linewidth=2)
    ax.fill_between(episodes, mean - std, mean + std, color=c, alpha=0.15)

# ── Styling ──────────────────────────────────────────────────────────────────
ax.set_xlabel("Episode", fontsize=13)
ax.set_ylabel("Cumulative Reward", fontsize=13)
ax.set_title("Cumulative Reward: QRC vs QC vs Q-Learning", fontsize=14, fontweight="bold")
ax.legend(fontsize=12, framealpha=0.9)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.grid(True, linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = "/mnt/user-data/outputs/cumulative_reward.png"
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
