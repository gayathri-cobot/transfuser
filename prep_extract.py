"""Stage 1 of the parallel preprocessing pipeline: extract a single bag run.

Does everything data_preprocess_distance.process_data() does *except* the
segmentation pass, so many copies of this can run concurrently without each one
loading its own SegFormer onto the GPU. Segmentation is handled once, batched,
by prep_segment.py.

Usually driven by run_prep.py, but standalone-usable:
    python prep_extract.py /workspace/bag_data/scenario_1/corridor1_w1_route1_225231
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_preprocess_distance as dp

# Defaults mirror data_preprocess_distance.main(), so running a bag through this
# pipeline and running it through that script directly produce the same frames.
DEFAULT_MAX_SYNC_DT = 0.1
DEFAULT_MIN_DIST = 0.05


def extract_bag(bag_path, output_path=None, max_sync_dt_s=DEFAULT_MAX_SYNC_DT,
                min_dist_m=DEFAULT_MIN_DIST):
    """Write the synced rgb/depth/lidar/costmap/trajectory streams for one bag."""
    bag_path = os.path.abspath(bag_path.rstrip('/'))
    output_path = output_path or dp._default_output_path(bag_path)
    os.makedirs(output_path, exist_ok=True)

    counts = dp._bag_topic_counts(bag_path)
    type_map = dp._type_map(dp._open_reader(bag_path))
    dp.save_synced_frames(bag_path, type_map, counts, output_path,
                          max_dt_ns=int(max_sync_dt_s * 1e9),
                          min_dist_m=min_dist_m)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', help='Path to a bag run directory (contains metadata.yaml + .mcap).')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: derived from the bag path, '
                             'matching data_preprocess_distance.py).')
    parser.add_argument('--max-sync-dt', type=float, default=DEFAULT_MAX_SYNC_DT,
                        help='Max allowed time gap, in seconds, between the reference '
                             'tf tick and every other synced stream '
                             '(default: %(default)s).')
    parser.add_argument('--min-dist', type=float, default=DEFAULT_MIN_DIST,
                        help='Minimum distance, in meters, the robot must travel '
                             '(map frame, base_link) between saved reference ticks '
                             '(default: %(default)s).')
    args = parser.parse_args()

    # OpenCV spins up a thread pool per process; with N workers on N cores that
    # is pure contention, so pin it to one thread and let run_prep.py own the
    # process-level parallelism.
    cv2.setNumThreads(1)

    output_path = extract_bag(args.bag, args.output, args.max_sync_dt, args.min_dist)
    # run_prep.py reads this line back to find the segmentation input.
    print(f"[info] output_path={output_path}")


if __name__ == '__main__':
    main()
