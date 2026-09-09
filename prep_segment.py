"""Stage 2 of the parallel preprocessing pipeline: batched floor segmentation.

Equivalent to data_preprocess_distance.save_segmented_images(), but over many output
directories at once and with the three things that actually cost wall-clock
fixed:

  * batched forward passes instead of one image per pass
  * fp16 autocast under inference_mode on CUDA
  * PIL decode + preprocess moved off the GPU critical path into DataLoader
    workers

The model is loaded once for the whole corpus, which is why stage 1 must not
run segmentation itself.

    python prep_segment.py /workspace/data/scenario_1/run_a /workspace/data/scenario_1/run_b
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL_ID = 'nvidia/segformer-b0-finetuned-ade-512-512'

# ADE20K class 3 = "floor" (confirmed via model.config.id2label; this checkpoint
# was trained on the 150-class ADE20K label set).
FLOOR_CLASS_ID = 3


class _FrontRgbDataset(torch.utils.data.Dataset):
    """Yields (preprocessed pixel_values, native (h, w), destination png path)."""

    def __init__(self, items, image_processor):
        self.items = items
        self.proc = image_processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        src, dst = self.items[i]
        img = Image.open(src).convert('RGB')
        pixel_values = self.proc(images=img, return_tensors='pt').pixel_values[0]
        w, h = img.size  # PIL .size is (W, H)
        return pixel_values, (h, w), dst


def _collate(batch):
    return (torch.stack([b[0] for b in batch]),
            [b[1] for b in batch],
            [b[2] for b in batch])


def _collect_items(output_paths, overwrite):
    """Build the (src, dst) work list across every run directory."""
    items, n_skipped = [], 0
    for output_path in output_paths:
        rgb_dir = os.path.join(output_path, 'rgb', 'front')
        segmented_dir = os.path.join(output_path, 'segmented', 'front')
        srcs = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        if not srcs:
            print(f"[warn] no front rgb frames under {rgb_dir}")
            continue
        os.makedirs(segmented_dir, exist_ok=True)
        for src in srcs:
            dst = os.path.join(segmented_dir, os.path.basename(src))
            if not overwrite and os.path.exists(dst):
                n_skipped += 1
                continue
            items.append((src, dst))
    if n_skipped:
        print(f"[info] skipping {n_skipped} frames that already have masks "
              f"(pass --overwrite to redo them)")
    return items


def segment(output_paths, batch_size=16, num_workers=4, overwrite=False):
    items = _collect_items(output_paths, overwrite)
    if not items:
        print("[info] nothing to segment")
        return 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(device).eval()

    loader = torch.utils.data.DataLoader(
        _FrontRgbDataset(items, image_processor),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=(device == 'cuda'),
    )

    print(f"[info] segmenting {len(items)} frames on {device} "
          f"(batch={batch_size}, workers={num_workers})")

    n_saved = 0
    for pixel_values, sizes, dsts in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        with torch.inference_mode():
            if device == 'cuda':
                with torch.autocast('cuda', dtype=torch.float16):
                    outputs = model(pixel_values=pixel_values)
            else:
                outputs = model(pixel_values=pixel_values)

        # post_process upsamples logits back to each frame's native resolution;
        # do that in fp32 so the bilinear interpolate matches the serial version.
        outputs.logits = outputs.logits.float()
        maps = image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=sizes)

        for pred, dst in zip(maps, dsts):
            mask = (pred == FLOOR_CLASS_ID).to(torch.uint8).mul_(255)
            cv2.imwrite(dst, mask.cpu().numpy(), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            n_saved += 1

        if n_saved % (batch_size * 20) < batch_size:
            print(f"[info] {n_saved}/{len(items)} masks written", flush=True)

    print(f"[info] saved {n_saved}/{len(items)} segmented images")
    return n_saved


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output_paths', nargs='+',
                        help='One or more run output directories (each containing rgb/front/).')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers for PIL decode + preprocess '
                             '(default: %(default)s).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-segment frames that already have a mask.')
    args = parser.parse_args()

    cv2.setNumThreads(1)  # the DataLoader workers already saturate the cores
    segment(args.output_paths, args.batch_size, args.num_workers, args.overwrite)


if __name__ == '__main__':
    main()
