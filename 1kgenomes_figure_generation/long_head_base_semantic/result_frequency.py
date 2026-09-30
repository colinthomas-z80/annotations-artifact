#! /usr/bin/env python

'''Given a set of task ids, read transaction logs and plot completion intervals over runtime.'''


import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sys

category_dict = {}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot task completion frequency for one or more transaction logs.'
    )
    parser.add_argument('num_logfiles', type=int, help='Number of logfile/label pairs')
    parser.add_argument(
        'inputs',
        nargs='+',
        help='Logfile paths followed by labels (num_logfiles each)',
    )
    parser.add_argument(
        '--subgraphs',
        action='store_true',
        help='Plot each logfile in a separate subplot',
    )
    parser.add_argument(
        '--output',
        default='result_frequency.png',
        help='Output image path (default: result_frequency.png)',
    )

    args = parser.parse_args()
    expected = args.num_logfiles * 2
    if len(args.inputs) != expected:
        parser.error(
            f'Expected {expected} values after num_logfiles: '
            f'{args.num_logfiles} logfiles then {args.num_logfiles} labels. '
            f'Got {len(args.inputs)}.'
        )

    args.logfiles = args.inputs[:args.num_logfiles]
    args.labels = args.inputs[args.num_logfiles:]
    return args


def compute_event_curve(logfile, task_ids):
    global category_dict
    events = [0]
    starttime = None

    with open(logfile, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('#'):
            continue

        parts = line.split()
        if not parts:
            continue

        if starttime is None:
            if parts[4] == "RUNNING":
                starttime = float(parts[0])
            continue

        donestr = ' '.join(parts[2:5])
        for task_id in task_ids:
            if donestr == f'TASK {task_id} DONE':
                events.append(float(parts[0]) - starttime)
                break

    if starttime is None:
        return np.array([0.0]), np.array([0.0]), 0.0

    endtime = float(lines[-1].split()[0]) - starttime
    events_s = np.array([e / 1000000 for e in events], dtype=float)
    endtime_s = endtime / 1000000

    x = np.linspace(0, endtime_s, 1000)
    y = np.zeros_like(x)

    for i in range(len(events_s) - 1):
        t_start = events_s[i]
        t_end = events_s[i + 1]
        period = t_end - t_start
        if period <= 0:
            continue

        mask = (x >= t_start) & (x <= t_end)
        x_interval = x[mask] 
        phase = 2 * np.pi * (x_interval - t_start) / period
        y[mask] = np.sin(phase)

    return x, events_s, y, endtime_s


def allan_variance_profile(signal):
    signal = np.asarray(signal, dtype=float)
    if signal.size < 3:
        return np.array([], dtype=float), np.array([], dtype=float)

    max_m = signal.size // 2
    if max_m < 1:
        return np.array([], dtype=float), np.array([], dtype=float)

    min_m = 4 if max_m >= 4 else 1
    n_points = min(20, max_m)
    m_values = np.unique(
        np.logspace(np.log10(min_m), np.log10(max_m), num=n_points).astype(int)
    )

    ms = []
    avars = []
    for m in m_values:
        k = signal.size // m
        if k < 2:
            continue

        # Standard Allan variance: 1/2 * mean((ybar_{i+1} - ybar_i)^2)
        # where ybar_i are averages over m-sample clusters.
        clustered = signal[:k * m].reshape(k, m).mean(axis=1)
        diff = np.diff(clustered)
        if diff.size == 0:
            continue
        avar = np.mean(diff ** 2) / 2.0
        ms.append(m)
        avars.append(avar)

    if not ms:
        return np.array([], dtype=float), np.array([], dtype=float)

    sample_counts = np.asarray(ms, dtype=float)
    avars = np.asarray(avars, dtype=float)
    return sample_counts, avars


def allan_deviation_profile(signal):
    sample_counts, avars = allan_variance_profile(signal)
    if sample_counts.size == 0:
        return sample_counts, np.array([], dtype=float)
    return sample_counts, np.sqrt(avars)

if __name__ == '__main__':
    args = parse_args()

    

    if args.subgraphs:
        fig, axes = plt.subplots(args.num_logfiles, 1, figsize=(12, 3 * args.num_logfiles), sharex=True)
        if args.num_logfiles == 1:
            axes = [axes]
        fig.suptitle('Inference Events')
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title('Inference Events')
        axes = [ax] * args.num_logfiles

    colors = cm.get_cmap('tab10', 10)

    tidfiles = ['tids_genomes_freq', 'tids_genomes_freq']
    allan_series = []
    for idx, (logfile, label) in enumerate(zip(args.logfiles, args.labels)):
        task_ids = [int(tid) for tid in open(tidfiles[idx]).read().splitlines() if tid.strip()]
        num_events = len(task_ids)

        x, events_s, y, endtime_s = compute_event_curve(logfile, task_ids)
        ax = axes[idx]
        color = colors(idx)

        period_uniform = endtime_s / num_events if num_events > 0 else endtime_s
        y_uniform = np.sin(2 * np.pi * x / period_uniform) if period_uniform > 0 else np.zeros_like(x)
        ax.plot(x, y_uniform, linewidth=1.5, color=color, alpha=0.3)

        ax.plot(x, y, linewidth=2, label=label, color=color)
        ax.scatter(events_s, [0.0 for _ in events_s], s=100, zorder=5, color=colors(idx+1))
        ax.grid(True, alpha=0.3)
        ax.axes.get_yaxis().set_visible(False)

        # sample_counts, adevs = allan_deviation_profile(y)
        # allan_series.append((label, color, sample_counts, adevs))
        # representative_adev = adevs[-1] if adevs.size > 0 else 0.0
        # ax.text(
        #     0.02,
        #     0.98,
        #     f'Allan deviation\n{representative_adev:.3e}',
        #     transform=ax.transAxes,
        #     va='top',
        #     ha='left',
        #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='0.7'),
        # )

        if args.subgraphs:
            ax.set_ylabel(label, rotation=0, labelpad=35, va='center')
            ax.legend(loc='upper right')

    if not args.subgraphs:
        axes[0].legend()

    axes[-1].set_xlabel('Time (s)')

    fig.tight_layout()
    plt.savefig(args.output, dpi=150)

    fig_allan, ax_allan = plt.subplots(figsize=(10, 6))
    for label, color, sample_counts, adevs in allan_series:
        if sample_counts.size == 0:
            continue
        ax_allan.loglog(sample_counts, adevs, linewidth=2, color=color, label=label)

    ax_allan.set_title('Allan Deviation vs Number of Samples')
    ax_allan.set_xlabel('Number of samples (m)')
    ax_allan.set_ylabel('Allan deviation')
    ax_allan.grid(True, which='both', alpha=0.3)
    ax_allan.legend()
    fig_allan.tight_layout()

    plt.savefig(args.output.replace('.png', '_allan.png'), dpi=150)
    plt.show()



    

    
            
