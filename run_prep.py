"""Two-stage parallel driver for data_preprocess_distance.py.

Stage 1  extraction   N independent processes, one bag each, no GPU involved.
Stage 2  segmentation one process, batched over every frame stage 1 produced.

Splitting them is the point: bag extraction is CPU/IO-bound and scales with
processes, while segmentation is GPU-bound and wants one process with big
batches. Running them together (as process_data does) gives you N CUDA contexts
serializing on one GPU.

    python run_prep.py /workspace/bag_data --jobs 4
    python run_prep.py /workspace/bag_data --stage extract --jobs 8
    python run_prep.py /workspace/bag_data --stage segment --batch-size 32
"""
import argparse
import concurrent.futures
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BAG_ROOT = '/workspace/bag_data'


def _find_bags(paths):
    """Resolve CLI paths to bag run directories.

    Each path may be a run directory itself (contains metadata.yaml) or a root to
    search recursively. Accepting explicit run dirs is what lets the S3 driver
    hand over exactly the window it has on local disk.
    """
    bags = []
    for path in paths:
        path = path.rstrip('/')
        if os.path.isfile(os.path.join(path, 'metadata.yaml')):
            bags.append(os.path.abspath(path))
            continue
        pattern = os.path.join(path, '**', 'metadata.yaml')
        bags.extend(os.path.abspath(os.path.dirname(p))
                    for p in glob.glob(pattern, recursive=True))
    return sorted(dict.fromkeys(bags))  # dedupe, keep deterministic order


def _output_path_for(bag_path):
    # Imported lazily so --help doesn't pay for torch/rosbag2, and reused rather
    # than reimplemented so the layout stays in sync with
    # data_preprocess_distance.py.
    sys.path.insert(0, HERE)

    import data_preprocess_distance as dp
    return dp._default_output_path(bag_path)


def _read_done(done_file):
    if not os.path.exists(done_file):
        return set()
    with open(done_file) as fh:
        return {line.strip() for line in fh if line.strip()}


def _extract_one(bag_path, log_dir, max_sync_dt, min_dist):
    """Run prep_extract.py for one bag in its own process. Returns (bag, ok)."""
    log_path = os.path.join(log_dir, os.path.basename(bag_path) + '.log')

    env = dict(os.environ)
    # One math thread per worker: N processes x M internal threads on N cores is
    # contention, not parallelism.
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    # Stage 1 never loads the model; make it impossible for it to grab the GPU.
    env['CUDA_VISIBLE_DEVICES'] = ''

    cmd = [sys.executable, os.path.join(HERE, 'prep_extract.py'), bag_path,
           '--max-sync-dt', str(max_sync_dt), '--min-dist', str(min_dist)]
    with open(log_path, 'w') as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return bag_path, proc.returncode == 0, log_path


def stage_extract(bags, jobs, log_dir, max_sync_dt, min_dist, done_file, resume):
    os.makedirs(log_dir, exist_ok=True)
    already = _read_done(done_file) if resume else set()
    todo = [b for b in bags if b not in already]
    if already:
        print(f"[info] resume: skipping {len(bags) - len(todo)} already-extracted bags")
    if not todo:
        return [], []

    print(f"[info] stage 1: extracting {len(todo)} bags with {jobs} processes "
          f"(logs -> {log_dir})")
    t0 = time.monotonic()
    ok, failed = [], []

    # Threads here only wait on subprocesses, so the GIL is irrelevant; the real
    # parallelism is the N child interpreters.
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_extract_one, b, log_dir, max_sync_dt, min_dist)
                   for b in todo]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            bag_path, succeeded, log_path = future.result()
            if succeeded:
                ok.append(bag_path)
                with open(done_file, 'a') as fh:
                    fh.write(bag_path + '\n')
            else:
                failed.append(bag_path)
                print(f"[warn] FAILED {bag_path} -- see {log_path}")
            print(f"[info] {i}/{len(todo)} bags done ({os.path.basename(bag_path)})",
                  flush=True)

    print(f"[info] stage 1 finished in {time.monotonic() - t0:.0f}s: "
          f"{len(ok)} ok, {len(failed)} failed")
    return ok, failed


def stage_segment(bags, batch_size, num_workers, overwrite):
    output_paths = [p for p in (_output_path_for(b) for b in bags)
                    if os.path.isdir(os.path.join(p, 'rgb', 'front'))]
    if not output_paths:
        print("[warn] stage 2: no run directories with front rgb frames; nothing to do")
        return

    print(f"[info] stage 2: segmenting {len(output_paths)} run directories")
    t0 = time.monotonic()

    cmd = [sys.executable, os.path.join(HERE, 'prep_segment.py'),
           '--batch-size', str(batch_size), '--num-workers', str(num_workers)]
    if overwrite:
        cmd.append('--overwrite')
    cmd += output_paths

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"[warn] stage 2 exited with code {proc.returncode}")
    print(f"[info] stage 2 finished in {time.monotonic() - t0:.0f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag_paths', nargs='*', default=[DEFAULT_BAG_ROOT],
                        help='Bag run directories, and/or roots to search for them '
                             '(default: %(default)s).')
    parser.add_argument('--jobs', '-j', type=int, default=4,
                        help='Concurrent extraction processes. Memory-bound, not '
                             'core-bound: each holds a tf buffer plus a handful of '
                             'undecoded full-res image messages (default: %(default)s).')
    parser.add_argument('--stage', choices=('all', 'extract', 'segment'), default='all')
    parser.add_argument('--max-sync-dt', type=float, default=0.1,
                        help='Passed through to save_synced_frames (default: %(default)s).')
    parser.add_argument('--min-dist', type=float, default=0.05,
                        help='Minimum distance, in meters, the robot must travel between '
                             'saved frames. Passed through to save_synced_frames '
                             '(default: %(default)s).')
    parser.add_argument('--log-dir', default='/tmp/prep_logs',
                        help='Per-bag stage 1 logs (default: %(default)s). Without this '
                             'the four processes interleave their prints.')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Stage 2 GPU batch size (default: %(default)s).')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Stage 2 DataLoader workers (default: %(default)s).')
    parser.add_argument('--resume', action='store_true',
                        help='Skip bags recorded as extracted in <log-dir>/prep_done.txt.')
    parser.add_argument('--overwrite-masks', action='store_true',
                        help='Re-segment frames that already have a mask.')
    args = parser.parse_args()

    bags = _find_bags(args.bag_paths)
    if not bags:
        sys.exit("[error] no bag runs (metadata.yaml) found under "
                 + ', '.join(args.bag_paths))
    print(f"[info] found {len(bags)} bag runs")

    os.makedirs(args.log_dir, exist_ok=True)
    done_file = os.path.join(args.log_dir, 'prep_done.txt')

    failed = []
    if args.stage in ('all', 'extract'):
        _, failed = stage_extract(bags, args.jobs, args.log_dir, args.max_sync_dt,
                                  args.min_dist, done_file, args.resume)

    if args.stage in ('all', 'segment'):
        # Segment everything that has frames on disk, including bags extracted by
        # an earlier run, but skip bags that just failed.
        stage_segment([b for b in bags if b not in failed],
                      args.batch_size, args.num_workers, args.overwrite_masks)

    if failed:
        print(f"[warn] {len(failed)} bags failed extraction:")
        for bag_path in failed:
            print(f"         {bag_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
