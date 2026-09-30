#! /usr/bin/env python

'''Plot the relative frequency deviation from the base period identified in result_frequency.py.'''

import argparse
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot the relative frequency deviation against the runtime baseline.'
    )
    parser.add_argument('num_logfiles', type=int, help='Number of logfile/label pairs')
    parser.add_argument(
        'inputs',
        nargs='+',
        help='Logfile paths followed by labels (num_logfiles each)',
    )
    parser.add_argument(
        '--taskfiles',
        nargs='+',
        default=None,
        help='Optional task-id files, one per logfile. Defaults to the same taskfiles used in result_frequency.py.',
    )
    parser.add_argument(
        '--base-frequency',
        type=float,
        default=None,
        help='Override the baseline frequency in Hz; defaults to the result_frequency.py estimate for each run.',
    )
    parser.add_argument(
        '--subgraphs',
        action='store_true',
        help='Plot each logfile in a separate subplot',
    )
    parser.add_argument(
        '--output',
        default='resource_objective.png',
        help='Output image path (default: resource_objective.png)',
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

    if args.taskfiles is not None:
        if len(args.taskfiles) != args.num_logfiles:
            parser.error(f'Expected {args.num_logfiles} taskfiles, got {len(args.taskfiles)}.')
        args.taskfiles = list(args.taskfiles)
    else:
        args.taskfiles = [None] * args.num_logfiles

    return args


def load_task_ids(taskfile):
    if taskfile is None:
        return []
    with open(taskfile, 'r') as f:
        return [int(tid) for tid in f.read().splitlines() if tid.strip()]


def parse_completion_times(logfile, task_ids):
    task_set = set(task_ids)
    starttime = None
    events = []

    with open(logfile, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            ts = float(parts[0])

            if starttime is None:
                if parts[2] == 'TASK' and parts[4] == 'RUNNING':
                    starttime = ts
                continue

            if parts[2] != 'TASK':
                continue

            try:
                tid = int(parts[3])
            except ValueError:
                continue

            if tid not in task_set:
                continue

            if parts[4] == 'DONE':
                events.append((ts - starttime) / 1_000_000.0)

    return np.asarray(events, dtype=float)


def estimate_base_frequency(event_times):
    if event_times.size < 2:
        return 0.0

    runtime = float(event_times[-1] - event_times[0])
    if runtime <= 0:
        return 0.0

    # Matches result_frequency.py's implied base frequency: counts / total runtime.
    return float(event_times.size) / runtime


def frequency_residual(event_times, base_frequency):
    if event_times.size == 0 or base_frequency <= 0:
        return np.array([]), np.array([])

    # Count the cumulative deficit relative to the expected number of completions at
    # each runtime point. This preserves startup gaps such as a run that begins with
    # no events for multiple baseline periods.
    x = np.concatenate(([0.0], event_times))
    observed = np.arange(x.size, dtype=float)
    expected = base_frequency * x
    residual = observed - expected
    return x, residual


if __name__ == '__main__':
    args = parse_args()

    if args.subgraphs:
        fig, axes = plt.subplots(args.num_logfiles, 1, figsize=(12, 3 * args.num_logfiles), sharex=False)
        if args.num_logfiles == 1:
            axes = [axes]
        fig.suptitle('Deviation from Average Throughput')
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title('Deviation from Average Throughput')
        axes = [ax] * args.num_logfiles

    all_ys = []
    for idx, (logfile, label) in enumerate(zip(args.logfiles, args.labels)):
        task_ids = load_task_ids(args.taskfiles[idx])
        event_times = parse_completion_times(logfile, task_ids)

        if event_times.size < 2:
            continue

        base_frequency = args.base_frequency
        if base_frequency is None:
            base_frequency = estimate_base_frequency(event_times)

        _, y = frequency_residual(event_times, base_frequency)
        if y.size > 0:
            all_ys.append(y)

    common_ymax = max((np.max(np.abs(y)) for y in all_ys), default=1.0)
    common_ymax = max(common_ymax, 1.0)

    for idx, (logfile, label) in enumerate(zip(args.logfiles, args.labels)):
        task_ids = load_task_ids(args.taskfiles[idx])
        event_times = parse_completion_times(logfile, task_ids)
        ax_i = axes[idx]

        if event_times.size < 2:
            ax_i.text(0.5, 0.5, 'No usable completion events', ha='center', va='center', transform=ax_i.transAxes)
            ax_i.set_xlabel('Runtime (s)')
            ax_i.set_ylabel('Deviation from base')
            ax_i.axhline(0, color='black', linewidth=1)
            ax_i.set_ylim(-common_ymax - 0.1, common_ymax + 0.1)
            continue

        base_frequency = args.base_frequency
        if base_frequency is None:
            base_frequency = estimate_base_frequency(event_times)

        x, y = frequency_residual(event_times, base_frequency)
        if x.size == 0:
            ax_i.text(0.5, 0.5, 'No valid residual values', ha='center', va='center', transform=ax_i.transAxes)
            ax_i.set_xlabel('Runtime (s)')
            ax_i.set_ylabel('Deviation from base')
            ax_i.axhline(0, color='black', linewidth=1)
            ax_i.set_ylim(-common_ymax - 0.1, common_ymax + 0.1)
            continue

        ax_i.plot(x, y, linewidth=2, label=label)
        cumulative_total = float(np.sum(y))
        box_y = 0.98 if args.subgraphs else 0.98 - idx * 0.08
        ax_i.text(
            0.02,
            box_y,
            f'{label}: final sum = {cumulative_total:.3f}',
            transform=ax_i.transAxes,
            va='top',
            ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'),
        )
        ax_i.axhline(0, color='black', linewidth=1)
        ax_i.set_xlabel('Runtime (s)')
        ax_i.set_ylabel('Relative deviation')
        #ax_i.set_title(f'{label} relative to {base_frequency:.3f} Hz base')
        ax_i.grid(True, alpha=0.3)
        ax_i.set_ylim(-common_ymax - 0.1, common_ymax + 0.1)

        if args.subgraphs:
            ax_i.legend(loc='upper right')

    if not args.subgraphs:
        axes[0].legend()

    fig.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.show()
