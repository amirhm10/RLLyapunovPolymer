from contextlib import contextmanager

import matplotlib as mpl


PAPER_COLORS = {
    "output": "tab:blue",
    "setpoint": "tab:red",
    "target": "tab:purple",
    "input_0": "tab:green",
    "input_1": "tab:orange",
    "reward": "tab:purple",
    "dist_qi": "tab:blue",
    "dist_qs": "tab:orange",
    "dist_ha": "tab:green",
    "state": "tab:blue",
    "disturbance": "tab:orange",
}


PAPER_RCPARAMS = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 2.0,
}


@contextmanager
def paper_plot_context():
    with mpl.rc_context(rc=PAPER_RCPARAMS):
        yield

