"""
plot_attack_path.py
────────────────────────────────────────────────────────────────
Two-panel visualization:
  LEFT   — Full enterprise network topology (static reference view)
  RIGHT  — Same graph with the trained agent's attack path overlaid,
           nodes colour-coded by access level, IDS alerts highlighted

Also includes:
  plot_training_curves()   — reads Monitor CSV logs and renders
                             reward / episode-length charts with
                             smoothed trend lines
"""

import glob
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from env.network_env import AccessLevel, RedTeamEnv
from env.network_topology import (
    NUM_NODES,
    SUBNETS,
    TARGET_NODE,
    build_network_graph,
    NODE_REGISTRY,
)

# ──────────────────────────────────────────────────────────────
# Theme
# ──────────────────────────────────────────────────────────────
BG_DARK  = "#0d1117"     # outer figure
BG_PANEL = "#161b22"     # axes background
GRID_CLR = "#21262d"

SUBNET_COLOR = {
    "DMZ":        "#ff6b6b",
    "FIREWALL":   "#ffd93d",
    "CORPORATE":  "#6bcb77",
    "DATACENTER": "#4d96ff",
}

ACCESS_COLOR = {
    AccessLevel.ROOT: "#ff4757",
    AccessLevel.USER: "#ffa502",
    "discovered":     "#5352ed",
    "unknown":        "#2f3542",
}

BASE_DIR = Path(__file__).parent.parent


# ──────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────
def _compute_layout() -> Dict[int, tuple]:
    """
    Fixed column layout: each subnet gets its own x-column so the
    left-to-right attack progression is visually obvious.
    """
    x_col = {"DMZ": 0.10, "FIREWALL_1": 0.32, "CORPORATE": 0.55,
              "FIREWALL_2": 0.77, "DATACENTER": 0.95}

    y_spread = {
        0: (x_col["DMZ"],        0.65),   # dmz-webserver
        1: (x_col["DMZ"],        0.35),   # dmz-dns
        2: (x_col["FIREWALL_1"], 0.50),   # fw-dmz-corp
        3: (x_col["CORPORATE"],  0.72),   # corp-ws-1
        4: (x_col["CORPORATE"],  0.50),   # corp-ws-2
        5: (x_col["CORPORATE"],  0.28),   # corp-webapp
        6: (x_col["FIREWALL_2"], 0.50),   # fw-corp-dc
        7: (x_col["DATACENTER"], 0.65),   # dc-database
        8: (x_col["DATACENTER"], 0.35),   # dc-active-directory
    }
    return y_spread


# ──────────────────────────────────────────────────────────────
# Topology panel (left)
# ──────────────────────────────────────────────────────────────
def _draw_topology(ax: plt.Axes, G: nx.DiGraph, pos: Dict):
    ax.set_facecolor(BG_PANEL)
    ax.set_title("Enterprise Network Topology", color="white",
                 fontsize=13, fontweight="bold", pad=14)
    ax.axis("off")

    # Subnet background blobs
    subnet_centres = {
        "DMZ":        (0.10, 0.50),
        "CORPORATE":  (0.55, 0.50),
        "DATACENTER": (0.95, 0.50),
    }
    for name, (cx, cy) in subnet_centres.items():
        ell = mpatches.Ellipse((cx, cy), 0.18, 0.55,
                                alpha=0.08, color=SUBNET_COLOR[name],
                                zorder=0)
        ax.add_patch(ell)
        ax.text(cx, cy + 0.33, name, ha="center", va="bottom",
                color=SUBNET_COLOR[name], fontsize=8, fontweight="bold")

    # Firewall markers
    for fx, label in [(0.32, "FW-1"), (0.77, "FW-2")]:
        ax.annotate("", xy=(fx, 0.80), xytext=(fx, 0.20),
                    arrowprops=dict(arrowstyle="-", color=SUBNET_COLOR["FIREWALL"],
                                    lw=2.5, linestyle="dashed"))
        ax.text(fx, 0.85, label, ha="center", color=SUBNET_COLOR["FIREWALL"],
                fontsize=8, fontweight="bold")

    # Edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#3a3f4b", arrows=True, arrowsize=10,
        width=0.9, alpha=0.7, connectionstyle="arc3,rad=0.06",
    )

    # Nodes
    colors = [SUBNET_COLOR[G.nodes[n]["subnet"]] for n in G.nodes]
    sizes  = [900 if n == TARGET_NODE else 550 for n in G.nodes]
    border = ["gold" if n == TARGET_NODE else "white" for n in G.nodes]
    lwidths= [3.0  if n == TARGET_NODE else 1.0  for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                           node_size=sizes, edgecolors=border,
                           linewidths=lwidths)

    # Short labels
    labels = {n: f"{n}\n{NODE_REGISTRY[n].name.split('-')[-1][:9]}"
              for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax,
                            font_size=6, font_color="white", font_weight="bold")

    # Legend
    patches = [mpatches.Patch(color=c, label=s)
               for s, c in SUBNET_COLOR.items()]
    patches.append(mpatches.Patch(color="gold", label="★ Target (AD)"))
    ax.legend(handles=patches, loc="lower left", facecolor=BG_DARK,
              labelcolor="white", fontsize=8, framealpha=0.9)


# ──────────────────────────────────────────────────────────────
# Attack-path overlay panel (right)
# ──────────────────────────────────────────────────────────────
def _draw_attack_path(ax: plt.Axes, G: nx.DiGraph, pos: Dict,
                      episode: Dict, title: str):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=14)
    ax.axis("off")

    access  = episode["access_level"]
    alerted = episode["alerted"]
    discov  = episode["discovered"]
    path    = episode["attack_path"]

    # Dim background edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#252c3a", arrows=True, arrowsize=6,
        width=0.5, alpha=0.35,
    )

    # Node colours by access level
    def _node_color(n):
        lv = int(access[n])
        if lv == AccessLevel.ROOT:    return ACCESS_COLOR[AccessLevel.ROOT]
        if lv == AccessLevel.USER:    return ACCESS_COLOR[AccessLevel.USER]
        if discov[n]:                 return ACCESS_COLOR["discovered"]
        return ACCESS_COLOR["unknown"]

    nc     = [_node_color(n) for n in G.nodes]
    sizes  = [900 if n == TARGET_NODE else 550 for n in G.nodes]
    border = ["gold" if n == TARGET_NODE
              else ("red" if alerted[n] else "#aabbcc")
              for n in G.nodes]
    lw     = [3.0 if n == TARGET_NODE
              else (2.0 if alerted[n] else 1.0)
              for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc,
                           node_size=sizes, edgecolors=border,
                           linewidths=lw)

    labels = {n: f"{n}\n{NODE_REGISTRY[n].name.split('-')[-1][:9]}"
              for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax,
                            font_size=6, font_color="white", font_weight="bold")

    # Attack-path edges with gradient colouring
    if len(path) > 1:
        path_edges  = list(zip(path[:-1], path[1:]))
        n_edges     = len(path_edges)
        palette     = plt.cm.YlOrRd(np.linspace(0.35, 0.95, n_edges))
        for i, (u, v) in enumerate(path_edges):
            if G.has_edge(u, v):
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)], ax=ax,
                    edge_color=[palette[i]], arrows=True,
                    arrowsize=22, width=3.5, alpha=0.95,
                    connectionstyle="arc3,rad=0.10",
                )

    # Step index annotations
    for idx, node in enumerate(path):
        x, y = pos[node]
        ax.text(x + 0.035, y + 0.055, f"S{idx}",
                color="white", fontsize=6.5, fontweight="bold",
                ha="center", va="center",
                bbox=dict(facecolor="#1c2333", alpha=0.7, pad=1.5,
                          boxstyle="round,pad=0.2"))

    # Stats box
    pwned = int(access[TARGET_NODE]) == AccessLevel.ROOT
    stats = (
        f"Steps      : {episode['steps']}\n"
        f"Reward     : {episode['total_reward']:.1f}\n"
        f"Discovered : {int(discov.sum())}/{NUM_NODES}\n"
        f"Owned      : {int((access > 0).sum())}/{NUM_NODES}\n"
        f"Root       : {int((access == 2).sum())}/{NUM_NODES}\n"
        f"IDS Alerts : {int(alerted.sum())}\n"
        f"Target     : {'✓ DOMAIN ADMIN' if pwned else '✗ Not reached'}"
    )
    ax.text(0.02, 0.02, stats, transform=ax.transAxes,
            fontsize=8, va="bottom", color="white", family="monospace",
            bbox=dict(facecolor="#0d1117", alpha=0.88, pad=7,
                      boxstyle="round,pad=0.4"))

    # Legend
    access_patches = [
        mpatches.Patch(color=ACCESS_COLOR[AccessLevel.ROOT], label="ROOT  access"),
        mpatches.Patch(color=ACCESS_COLOR[AccessLevel.USER], label="USER  access"),
        mpatches.Patch(color=ACCESS_COLOR["discovered"],      label="Discovered"),
        mpatches.Patch(color=ACCESS_COLOR["unknown"],         label="Unknown"),
        mpatches.Patch(color="red",                           label="⚠ IDS Alert"),
    ]
    ax.legend(handles=access_patches, loc="lower right",
              facecolor=BG_DARK, labelcolor="white",
              fontsize=8, framealpha=0.9)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def run_episode_with_model(model, deterministic: bool = True) -> Dict:
    """
    Execute one evaluation episode with the trained model.
    Returns all state arrays needed for visualisation.
    """
    env = RedTeamEnv(render_mode="human")
    obs, _ = env.reset()
    done   = False

    while not done:
        action_masks = env.action_masks()
    action, _ = model.predict(obs, deterministic=deterministic,
                              action_masks=action_masks)
    obs, _, terminated, truncated, _ = env.step(int(action))
    done = terminated or truncated

    return {
        "attack_path":  env.attack_path,
        "access_level": env.access_level.copy(),
        "alerted":      env.alerted.copy(),
        "discovered":   env.discovered.copy(),
        "total_reward": env.total_reward,
        "steps":        env.step_count,
    }


def plot_network_and_attack_path(
    episode_result: Dict,
    save_path: Optional[str] = None,
    title: str = "Trained RL Agent — Attack Path",
):
    """
    Render the two-panel topology + attack-path visualisation.
    Saves to save_path if provided; always calls plt.show().
    """
    G   = build_network_graph()
    pos = _compute_layout()

    fig, (ax_topo, ax_atk) = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor(BG_DARK)

    _draw_topology(ax_topo, G, pos)
    _draw_attack_path(ax_atk, G, pos, episode_result, title)

    plt.tight_layout(pad=2.5)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=BG_DARK)
        print(f"✓ Visualisation saved → {save_path}")

    plt.show()


def plot_training_curves(log_dir: Optional[str] = None):
    """
    Read Monitor CSV files from the training run and plot:
      - Episode reward (raw + smoothed)
      - Episode length (raw + smoothed)
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas required: pip install pandas")
        return

    log_dir = log_dir or str(BASE_DIR / "logs")
    files   = glob.glob(f"{log_dir}/monitor_*.monitor.csv")

    if not files:
        print(f"No monitor files in {log_dir}. Run training first.")
        return

    all_r, all_l = [], []
    for f in files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(f, comment="#", header=0)
        if {"r", "l"}.issubset(df.columns):
            all_r.extend(df["r"].tolist())
            all_l.extend(df["l"].tolist())

    if not all_r:
        print("Monitor files are empty — no episodes logged yet.")
        return

    rewards  = np.array(all_r)
    lengths  = np.array(all_l)
    episodes = np.arange(len(rewards))
    window   = max(len(rewards) // 40, 5)

    def smooth(arr, w):
        return np.convolve(arr, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(BG_DARK)
    fig.suptitle("Red-Team RL — Training Metrics", color="white",
                 fontsize=16, fontweight="bold")

    plots = [
        (axes[0, 0], rewards,         "Episode Reward (raw)",      "#ff6b6b"),
        (axes[0, 1], lengths,         "Episode Length (raw)",      "#4d96ff"),
        (axes[1, 0], smooth(rewards, window), "Smoothed Reward",   "#ffd93d"),
        (axes[1, 1], smooth(lengths, window), "Smoothed Length",   "#6bcb77"),
    ]

    for ax, data, label, color in plots:
        ax.set_facecolor(BG_PANEL)
        ax.plot(data, color=color, alpha=0.80, linewidth=0.9, label=label)

        # Trend line
        if len(data) > 10:
            x  = np.arange(len(data))
            z  = np.polyfit(x, data, 1)
            ax.plot(np.poly1d(z)(x), "--", color="white",
                    alpha=0.45, linewidth=1.5, label="Trend")

        ax.set_title(label, color="white", fontsize=11)
        ax.set_xlabel("Episode", color="white", fontsize=9)
        ax.set_ylabel(label.split(" ")[1], color="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.set_facecolor(BG_PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3f4b")
        ax.legend(facecolor=BG_DARK, labelcolor="white", fontsize=8)

    plt.tight_layout()
    out = str(BASE_DIR / "logs" / "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    print(f"✓ Training curves → {out}")
    plt.show()
