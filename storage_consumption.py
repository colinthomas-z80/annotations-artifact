#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Read a CSV with columns time, percent_complete, and per-worker disk usage, '
            'sum the worker columns, and plot the result over time.'
        )
    )
    parser.add_argument('csvfiles', type=Path, help='Input CSV file', nargs='+')
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=Path('storage_consumption.png'),
        help='Output image path (default: storage_consumption.png)',
    )
    parser.add_argument(
        '--title',
        default='Disk consumption over time',
        help='Title for the generated plot.',
    )

    parser.add_argument(
        '--subtract-subgraphs',
        type=Path,
        default=None,
        help='tidfile, debuglog to subtract from the main graph',
        nargs='+',
        )

    parser.add_argument(
        '--subgraph-storage-size',
        type=float,
        default=10,
        help='Storage size for each subgraph.',
    )

    parser.add_argument(
        '--labels',
        type=str,
        default=None,
        nargs='+',
        help='Labels for each CSV file in the plot legend.',
    )

    return parser.parse_args()


def load_usage_series(csv_path):
    with csv_path.open(newline='') as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError(f'{csv_path} does not contain a header and any data rows.')

    header = rows[0]
    if len(header) < 3:
        raise ValueError(
            'Expected at least three columns: time, percent_complete, and one or more disk usage values.'
        )

    times = []
    totals = []

    last_usage = np.zeros(len(header) - 2, dtype=float)
    for row in rows[1:]:
        if not row or all((cell or '').strip() == '' for cell in row):
            continue
        if len(row) < 3:
            continue

        try:
            time_value = float(row[0].strip())
        except ValueError:
            continue

        usage_values = last_usage.copy()
        for idx, cell in enumerate(row[2:]):
            text = (cell or '').strip()
            if text == '':
                continue
            #print(text)
            #print(usage_values)
            usage_values[idx] = float(text)
        last_usage = usage_values

        if not usage_values.any():
            continue

        times.append(time_value)
        totals.append(sum(usage_values))

    if not times:
        raise ValueError(f'No numeric disk usage values found in {csv_path}.')

    return np.asarray(times, dtype=float), np.asarray(totals, dtype=float)

def get_subgraph_completion_times(subgraph_paths):
    completion_times = []
    tidfile_path, debuglog_path = subgraph_paths
    tids = [id for id in tidfile_path.read_text().splitlines() if id.strip()]

    print(debuglog_path)

    debuglog_lines = debuglog_path.read_text().splitlines()
    starttime = None
    for tid in tids:
        for line in debuglog_lines:
            if starttime is None and 'manager start' in line:
                parts = line.split()
                starttime = parts[1].split(':')

            if f"Task {tid} state change: RETRIEVED (4) to DONE (5)" in line:
                parts = line.split()
                complete_time = parts[1]

                # convert hh:mm:ss.mmm to seconds
                h, m, s = complete_time.split(':')
                seconds = int(h) * 3600 + int(m) * 60 + float(s)
                h, m, s = starttime
                start_seconds = int(h) * 3600 + int(m) * 60 + float(s)
                seconds -= start_seconds
                completion_times.append(seconds)

                break
    return completion_times


def main():
    args = parse_args()

    fig, ax = plt.subplots(figsize=(10, 5))
    
    for idx, csvfile in enumerate(args.csvfiles):

        times, totals = load_usage_series(csvfile)

        #print(times[1])

        y_values = totals
        print()
        print(csvfile)

        if args.subtract_subgraphs:
            times_complete = get_subgraph_completion_times(args.subtract_subgraphs[2*idx:2*idx+2])

            storage_released = np.zeros_like(y_values)
            for t in times_complete:
                last_event = None
                for j, event in enumerate(times):
                    #print(f"Comparing subgraph completion time {t} with event time {event}")
                    if last_event is not None and last_event <= t < event:
                        following_events = y_values[j:]
                        storage_freed = np.zeros_like(following_events)
                        storage_freed[:] = args.subgraph_storage_size
                        y_values[j:] -= storage_freed
                        break
                    last_event = event

            #storage_released = np.cumsum(storage_released)
            #print(storage_released)
            #y_values -= storage_released

        ax.plot(times[:-5], y_values[:-5], linewidth=2, label=args.labels[idx])

    ax.set_xlabel('Time')
    ax.set_ylabel('Instantaneous Disk Consumption (GB)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.legend(ncol=len(args.csvfiles))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f'Saved plot to {args.output}')
    plt.close(fig)


if __name__ == '__main__':
    main()
