#!/usr/bin/env python3
"""
Standalone inference script for WS-Mask2Former model.
Validates INSTANCE_ON mode produces correct bounding box outputs.

Usage:
    python inference_server.py --image input.jpg --subnet max
    python inference_server.py --benchmark  # Test all subnets
    python inference_server.py --print-labels  # Print ADE20K class mapping
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Any

import cv2
import numpy as np

# Ensure mask2former is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks

from mask2former import add_maskformer2_config
from mask2former.engine.defaults import DefaultPredictor


# ADE20K class indices that map to Pylot's OBSTACLE_LABELS
# These need verification against actual metadata
ADE_TO_PYLOT_HARDCODED = {
    12: 'person',      # "person, individual, someone, somebody, mortal, soul"
    20: 'car',         # "car, auto, automobile, machine, motorcar"
    80: 'bus',         # Needs verification
    83: 'truck',       # Needs verification
    90: 'motorcycle',  # Needs verification
    127: 'bicycle',    # Needs verification
}

# Full name mappings for lookup
ADE_NAME_TO_PYLOT = {
    'person': 'person',
    'car': 'car',
    'bus': 'bus',
    'truck': 'truck',
    'motorcycle': 'motorcycle',
    'motorbike': 'motorcycle',
    'minibike': 'motorcycle',
    'bicycle': 'bicycle',
    'bike': 'bicycle',
}

# Pylot's valid labels
OBSTACLE_LABELS = {
    'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'vehicle', 'person'
}

# Subnet configurations matching test_ws_net.py
MODEL_NETWORK_CONFIGS = {
    # Note: backbone subnet requires SNNet; ResNet backbone is fixed
    "min": {
        "pixel_decoder": [0],
        "predictor": [0]
    },
    "middle": {
        "pixel_decoder": [1],
        "predictor": [1]
    },
    "max": {
        "pixel_decoder": [2],
        "predictor": [2]
    }
}


def get_config(config_path: str, weights_path: str, device: str = "cuda:0") -> Any:
    """Load and configure the model."""
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    # Allow new config keys from newer detectron2 versions
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = device

    # Critical: Enable INSTANCE mode for per-object detection
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False

    cfg.freeze()
    return cfg


def build_label_map(metadata) -> Dict[int, str]:
    """
    Build mapping from ADE20K class indices to Pylot labels.
    Tries metadata first, falls back to hardcoded.
    """
    mapping = {}

    if metadata is not None:
        # Try stuff_classes (semantic segmentation)
        class_names = getattr(metadata, 'stuff_classes', None)
        if class_names is None:
            # Try thing_classes (instance segmentation)
            class_names = getattr(metadata, 'thing_classes', None)

        if class_names:
            for idx, name in enumerate(class_names):
                # Parse the ADE20K format: "car, auto, automobile, machine, motorcar"
                name_lower = name.lower().strip()
                # Check each word in the comma-separated name
                for word in name_lower.replace(',', ' ').split():
                    word = word.strip()
                    if word in ADE_NAME_TO_PYLOT:
                        mapping[idx] = ADE_NAME_TO_PYLOT[word]
                        break

    # If no mapping found from metadata, use hardcoded
    if not mapping:
        print("WARNING: Using hardcoded ADE20K label mapping")
        mapping = ADE_TO_PYLOT_HARDCODED.copy()

    return mapping


def print_ade20k_labels(metadata):
    """Print all ADE20K class labels with indices for verification."""
    print("\n" + "="*80)
    print("ADE20K CLASS LABELS")
    print("="*80)

    class_names = getattr(metadata, 'stuff_classes', None)
    source = "stuff_classes"
    if class_names is None:
        class_names = getattr(metadata, 'thing_classes', None)
        source = "thing_classes"

    if class_names is None:
        print("ERROR: No class names found in metadata!")
        return

    print(f"Source: {source}")
    print(f"Total classes: {len(class_names)}\n")

    # Find relevant classes
    relevant = []
    for idx, name in enumerate(class_names):
        name_lower = name.lower()
        for keyword in ['person', 'car', 'bus', 'truck', 'motorcycle', 'motorbike',
                       'bicycle', 'bike', 'vehicle', 'minibike']:
            if keyword in name_lower:
                relevant.append((idx, name))
                break

    print("RELEVANT CLASSES FOR AV DETECTION:")
    print("-" * 60)
    for idx, name in relevant:
        print(f"  [{idx:3d}] {name}")

    print("\n" + "-" * 60)
    print("ALL CLASSES:")
    print("-" * 60)
    for idx, name in enumerate(class_names):
        print(f"  [{idx:3d}] {name}")

    print("="*80 + "\n")


def set_subnet(model, subnet_name: str):
    """
    Activate a specific subnet configuration.

    The model has elastic depth in:
    - pixel_decoder.transformer (DEPTH_LIST = [0, 1, 2])
    - predictor (DEPTH_LIST = [0, 1, 2])

    Backbone is fixed (ResNet) - no subnet switching there.
    """
    subnet_cfg = MODEL_NETWORK_CONFIGS.get(subnet_name)
    if subnet_cfg is None:
        print(f"Unknown subnet: {subnet_name}. Available: {list(MODEL_NETWORK_CONFIGS.keys())}")
        return

    print(f"Setting subnet: {subnet_name} -> {subnet_cfg}")

    # Set pixel decoder transformer depth
    if 'pixel_decoder' in subnet_cfg:
        pd = model.sem_seg_head.pixel_decoder
        if hasattr(pd, 'transformer') and hasattr(pd.transformer, 'set_active_subnet'):
            pd.transformer.set_active_subnet(depth_list=subnet_cfg['pixel_decoder'])
            print(f"  pixel_decoder depth: {subnet_cfg['pixel_decoder']}")
        else:
            print(f"  WARNING: pixel_decoder subnet not available")

    # Set predictor (transformer decoder) depth
    if 'predictor' in subnet_cfg:
        pred = model.sem_seg_head.predictor
        if hasattr(pred, 'set_active_subnet'):
            pred.set_active_subnet(depth_list=subnet_cfg['predictor'])
            print(f"  predictor depth: {subnet_cfg['predictor']}")
        else:
            print(f"  WARNING: predictor subnet not available")

    # Note: backbone subnet switching requires SNNet, not available with ResNet
    if 'backbone' in subnet_cfg:
        if hasattr(model.backbone, 'set_active_subnet'):
            model.backbone.set_active_subnet(depth_list=subnet_cfg['backbone'])
            print(f"  backbone depth: {subnet_cfg['backbone']}")
        else:
            print(f"  Note: backbone is fixed (ResNet) - no subnet switching")


def run_inference(predictor, image: np.ndarray, label_map: Dict[int, str],
                  min_score: float = 0.3, min_area: float = 200,
                  all_labels: bool = False, metadata=None) -> List[Dict]:
    """
    Run inference and convert to bounding box detections.

    Returns list of detections in format:
    [{"label": "car", "score": 0.95, "bbox": [x1, y1, x2, y2]}, ...]
    """
    outputs = predictor(image)
    instances = outputs.get('instances', None)

    if instances is None:
        return []

    instances = instances.to('cpu')
    detections = []

    # Get or compute bounding boxes from masks
    # The model may output zero boxes even when masks exist
    if hasattr(instances, 'pred_masks') and len(instances) > 0:
        masks = instances.pred_masks
        if isinstance(masks, torch.Tensor):
            masks = (masks > 0.5).numpy()  # threshold masks
        bitmasks = BitMasks(masks)
        instances.pred_boxes = bitmasks.get_bounding_boxes()
    elif not hasattr(instances, 'pred_boxes') or instances.pred_boxes is None:
        return []

    num_instances = len(instances)
    for idx in range(num_instances):
        score = float(instances.scores[idx])
        if score < min_score:
            continue

        # Get bounding box [x1, y1, x2, y2]
        bbox = instances.pred_boxes.tensor[idx].numpy().tolist()
        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            continue
        if width * height < min_area:
            continue

        # Map class to Pylot label or get raw ADE20K label
        class_idx = int(instances.pred_classes[idx])

        if all_labels:
            # Get raw ADE20K label from metadata
            class_names = None
            if metadata is not None:
                class_names = getattr(metadata, 'stuff_classes', None)
                if class_names is None:
                    class_names = getattr(metadata, 'thing_classes', None)

            if class_names and class_idx < len(class_names):
                # Take first word of the ADE20K name (e.g., "car" from "car, auto, automobile")
                raw_name = class_names[class_idx]
                label = raw_name.split(',')[0].strip()
            else:
                label = f"class_{class_idx}"
        else:
            # Filter to Pylot labels only
            label = label_map.get(class_idx, None)
            if label is None or label not in OBSTACLE_LABELS:
                continue

        detections.append({
            "label": label,
            "score": round(score, 4),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "class_idx": class_idx  # Include for debugging
        })

    return detections


def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes on image."""
    result = image.copy()

    # Known colors for Pylot labels
    colors = {
        'person': (0, 255, 0),
        'car': (0, 0, 255),
        'bus': (255, 0, 0),
        'truck': (255, 128, 0),
        'motorcycle': (128, 0, 255),
        'bicycle': (0, 255, 255),
        'vehicle': (0, 0, 255),
    }

    # Generate consistent colors for other labels based on hash
    def get_color(label):
        if label in colors:
            return colors[label]
        # Generate color from label hash
        h = hash(label)
        return ((h * 37) % 256, (h * 73) % 256, (h * 113) % 256)

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        score = det['score']
        color = get_color(label)

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {score:.2f}"

        # Add background for better readability
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result, (x1, max(0, y1-th-5)), (x1+tw, max(0, y1-2)), (0, 0, 0), -1)
        cv2.putText(result, text, (x1, max(th, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return result


def benchmark_subnets(predictor, image: np.ndarray, label_map: Dict[int, str],
                      num_warmup: int = 3, num_runs: int = 10,
                      min_score: float = 0.3, min_area: float = 200,
                      all_labels: bool = False, metadata=None):
    """Benchmark inference latency across all subnets."""
    print("\n" + "="*80)
    print("SUBNET LATENCY BENCHMARK")
    print("="*80)

    results = {}

    for subnet_name in ["min", "middle", "max"]:
        print(f"\nTesting subnet: {subnet_name}")
        set_subnet(predictor.model, subnet_name)

        # Warmup
        for _ in range(num_warmup):
            _ = predictor(image)

        # Timed runs
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            outputs = predictor(image)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        avg_ms = np.mean(times)
        std_ms = np.std(times)

        # Also get detection count
        detections = run_inference(predictor, image, label_map, min_score, min_area,
                                   all_labels=all_labels, metadata=metadata)

        results[subnet_name] = {
            "avg_ms": round(avg_ms, 2),
            "std_ms": round(std_ms, 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "num_detections": len(detections)
        }

        print(f"  Latency: {avg_ms:.2f} +/- {std_ms:.2f} ms")
        print(f"  Range: [{min(times):.2f}, {max(times):.2f}] ms")
        print(f"  Detections: {len(detections)}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Subnet':<10} {'Avg (ms)':<12} {'Std (ms)':<12} {'Detections':<12}")
    print("-" * 50)
    for name, data in results.items():
        print(f"{name:<10} {data['avg_ms']:<12.2f} {data['std_ms']:<12.2f} {data['num_detections']:<12}")

    return results


def main():
    parser = argparse.ArgumentParser(description='WS-Mask2Former Inference Test')
    parser.add_argument('--config', default='sem_ade_output_2/config.yaml',
                        help='Path to config YAML')
    parser.add_argument('--weights', default='sem_ade_output_2/model_final.pth',
                        help='Path to model weights')
    parser.add_argument('--image', default='input.jpg',
                        help='Input image path')
    parser.add_argument('--output', default=None,
                        help='Output image path (optional)')
    parser.add_argument('--subnet', default='max',
                        choices=['min', 'middle', 'max'],
                        help='Subnet to use')
    parser.add_argument('--device', default='cuda:0',
                        help='Device (cuda:0, cuda:1, cpu)')
    parser.add_argument('--min-score', type=float, default=0.3,
                        help='Minimum detection score')
    parser.add_argument('--min-area', type=float, default=200,
                        help='Minimum bbox area in pixels')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark all subnets')
    parser.add_argument('--print-labels', action='store_true',
                        help='Print ADE20K class labels')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON only')
    parser.add_argument('--all-labels', action='store_true',
                        help='Show all ADE20K labels (not just Pylot-relevant)')

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config) if not os.path.isabs(args.config) else args.config
    weights_path = os.path.join(script_dir, args.weights) if not os.path.isabs(args.weights) else args.weights
    image_path = os.path.join(script_dir, args.image) if not os.path.isabs(args.image) else args.image

    # Check files exist
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)
    if not os.path.exists(weights_path):
        print(f"ERROR: Weights not found: {weights_path}")
        sys.exit(1)

    if not args.json:
        print(f"Config: {config_path}")
        print(f"Weights: {weights_path}")
        print(f"Device: {args.device}")

    # Load model
    cfg = get_config(config_path, weights_path, args.device)
    predictor = DefaultPredictor(cfg)

    # Get metadata
    metadata = None
    if len(cfg.DATASETS.TRAIN):
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    elif len(cfg.DATASETS.TEST):
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    # Print labels if requested
    if args.print_labels:
        print_ade20k_labels(metadata)
        if not args.image:
            return

    # Build label map
    label_map = build_label_map(metadata)
    if not args.json:
        print(f"\nLabel mapping ({len(label_map)} classes):")
        for idx, label in sorted(label_map.items()):
            print(f"  [{idx}] -> {label}")

    # Load image
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Failed to read image: {image_path}")
        sys.exit(1)

    if not args.json:
        print(f"\nImage: {image_path} ({image.shape[1]}x{image.shape[0]})")

    # Benchmark mode
    if args.benchmark:
        results = benchmark_subnets(predictor, image, label_map,
                                     min_score=args.min_score, min_area=args.min_area,
                                     all_labels=args.all_labels, metadata=metadata)
        if args.json:
            print(json.dumps(results, indent=2))
        return

    # Single inference mode
    set_subnet(predictor.model, args.subnet)

    # Warmup
    _ = predictor(image)

    # Timed inference
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    _ = predictor(image)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    latency_ms = (time.perf_counter() - start) * 1000

    # Get detections
    detections = run_inference(predictor, image, label_map,
                               args.min_score, args.min_area,
                               all_labels=args.all_labels, metadata=metadata)

    result = {
        "subnet": args.subnet,
        "latency_ms": round(latency_ms, 2),
        "num_detections": len(detections),
        "detections": detections
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nSubnet: {args.subnet}")
        print(f"Latency: {latency_ms:.2f} ms")
        print(f"Detections: {len(detections)}")
        print("\nDetailed results:")
        for i, det in enumerate(detections):
            print(f"  [{i}] {det['label']:<12} score={det['score']:.3f} "
                  f"bbox={det['bbox']} (class_idx={det['class_idx']})")

    # Save visualization if requested
    if args.output:
        annotated = draw_detections(image, detections)
        cv2.imwrite(args.output, annotated)
        if not args.json:
            print(f"\nSaved annotated image to: {args.output}")


if __name__ == '__main__':
    main()
