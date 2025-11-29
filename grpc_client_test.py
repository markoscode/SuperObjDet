#!/usr/bin/env python3
"""
Test client for the gRPC detection service.

Usage:
    python grpc_client_test.py --image input.jpg
    python grpc_client_test.py --image input.jpg --server localhost:50051
    python grpc_client_test.py --status  # Just check server status
"""

import argparse
import os
import sys
import time

import cv2
import grpc
import numpy as np

# Add this directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection_service import (
    detection_pb2,
    detection_pb2_grpc,
)


def test_status(stub):
    """Test GetStatus RPC."""
    print("\n[GetStatus]")
    response = stub.GetStatus(detection_pb2.StatusRequest())
    print(f"  Ready: {response.ready}")
    print(f"  Active subnet: {response.active_subnet}")
    print(f"  Available subnets: {list(response.available_subnets)}")
    print(f"  Device: {response.device}")
    print(f"  Weights: {response.weights_path}")
    return response


def test_detect(stub, image_path: str, subnet: str = "", min_score: float = 0.3,
                all_labels: bool = False):
    """Test Detect RPC."""
    print(f"\n[Detect] Image: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: Could not read image: {image_path}")
        return None

    height, width, channels = image.shape
    print(f"  Image size: {width}x{height}x{channels}")

    # Serialize image to bytes
    image_bytes = image.tobytes()

    # Create request
    request = detection_pb2.DetectRequest(
        image_data=image_bytes,
        width=width,
        height=height,
        channels=channels,
        subnet=subnet,
        min_score=min_score,
        all_labels=all_labels,
    )

    # Time the RPC call (includes network latency)
    rpc_start = time.perf_counter()
    response = stub.Detect(request)
    rpc_time_ms = (time.perf_counter() - rpc_start) * 1000

    print(f"  Success: {response.success}")
    if not response.success:
        print(f"  Error: {response.error}")
        return response

    print(f"  Subnet used: {response.subnet_used}")
    print(f"  Inference time: {response.inference_time_ms:.2f} ms")
    print(f"  Total server time: {response.total_time_ms:.2f} ms")
    print(f"  RPC round-trip: {rpc_time_ms:.2f} ms")
    print(f"  Network overhead: {rpc_time_ms - response.total_time_ms:.2f} ms")
    print(f"  Detections: {len(response.detections)}")

    for i, det in enumerate(response.detections):
        print(f"    [{i}] {det.label:<12} score={det.score:.3f} "
              f"bbox=[{det.x_min}, {det.y_min}, {det.x_max}, {det.y_max}]")

    return response


def test_set_subnet(stub, subnet: str):
    """Test SetSubnet RPC."""
    print(f"\n[SetSubnet] -> {subnet}")
    response = stub.SetSubnet(detection_pb2.SetSubnetRequest(subnet=subnet))
    print(f"  Success: {response.success}")
    if not response.success:
        print(f"  Error: {response.error}")
    print(f"  Active subnet: {response.active_subnet}")
    return response


def run_benchmark(stub, image_path: str, num_runs: int = 10):
    """Benchmark detection across subnets."""
    print(f"\n{'='*60}")
    print("BENCHMARK")
    print(f"{'='*60}")

    # Load image once
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image: {image_path}")
        return

    height, width, channels = image.shape
    image_bytes = image.tobytes()

    results = {}

    for subnet in ["min", "middle", "max"]:
        print(f"\n[Subnet: {subnet}]")
        test_set_subnet(stub, subnet)

        times = []
        for i in range(num_runs):
            request = detection_pb2.DetectRequest(
                image_data=image_bytes,
                width=width,
                height=height,
                channels=channels,
                min_score=0.3,
            )

            start = time.perf_counter()
            response = stub.Detect(request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

            if i == 0:
                num_detections = len(response.detections)

        avg_ms = np.mean(times)
        std_ms = np.std(times)
        results[subnet] = {
            "avg_rpc_ms": round(avg_ms, 2),
            "std_ms": round(std_ms, 2),
            "detections": num_detections,
        }
        print(f"  RPC latency: {avg_ms:.2f} +/- {std_ms:.2f} ms ({num_runs} runs)")
        print(f"  Detections: {num_detections}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Subnet':<10} {'Avg RPC (ms)':<15} {'Std (ms)':<12} {'Detections':<12}")
    print("-" * 50)
    for name, data in results.items():
        print(f"{name:<10} {data['avg_rpc_ms']:<15.2f} {data['std_ms']:<12.2f} {data['detections']:<12}")


def main():
    parser = argparse.ArgumentParser(description='Test gRPC detection service')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_image = os.path.join(script_dir, 'input.jpg')

    parser.add_argument('--server', default='localhost:50051',
                        help='Server address (default: localhost:50051)')
    parser.add_argument('--image', default=default_image,
                        help='Image to test with')
    parser.add_argument('--subnet', default='',
                        help='Subnet to use for detection (optional)')
    parser.add_argument('--min-score', type=float, default=0.3,
                        help='Minimum score threshold')
    parser.add_argument('--all-labels', action='store_true',
                        help='Show all ADE20K labels')
    parser.add_argument('--status', action='store_true',
                        help='Just get server status')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark across all subnets')
    parser.add_argument('--num-runs', type=int, default=10,
                        help='Number of runs for benchmark')

    args = parser.parse_args()

    # Connect to server
    print(f"Connecting to {args.server}...")
    channel = grpc.insecure_channel(args.server)
    stub = detection_pb2_grpc.DetectionServiceStub(channel)

    try:
        # Test status first
        test_status(stub)

        if args.status:
            return

        if args.benchmark:
            run_benchmark(stub, args.image, args.num_runs)
            return

        # Test detection
        if os.path.exists(args.image):
            test_detect(stub, args.image, args.subnet, args.min_score, args.all_labels)
        else:
            print(f"\nWARNING: Image not found: {args.image}")
            print("Use --image to specify an image path")

    except grpc.RpcError as e:
        print(f"\nRPC ERROR: {e.code()}: {e.details()}")
        sys.exit(1)
    finally:
        channel.close()


if __name__ == '__main__':
    main()
