#!/usr/bin/env python3
"""
gRPC server wrapper for WS-Mask2Former detection model.

This is an ADDITIVE layer on top of inference_server.py.
The original inference_server.py CLI remains fully functional.

Usage:
    python grpc_server.py --port 50051
    python grpc_server.py --port 50051 --subnet middle --device cuda:0
"""

import argparse
import os
import sys
import time
from concurrent import futures
from typing import Optional

import grpc
import numpy as np

# Add this directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the existing inference_server module (NO modifications to it)
from inference_server import (
    get_config,
    build_label_map,
    set_subnet,
    run_inference,
    MODEL_NETWORK_CONFIGS,
    OBSTACLE_LABELS,
)

# Import generated gRPC stubs
from detection_service import (
    detection_pb2,
    detection_pb2_grpc,
)


class DetectionServicer(detection_pb2_grpc.DetectionServiceServicer):
    """
    gRPC service implementation for object detection.
    Wraps the inference_server.py functions.
    """

    def __init__(self, config_path: str, weights_path: str, device: str = "cuda:0",
                 initial_subnet: str = "max"):
        """
        Initialize the detection service.

        Args:
            config_path: Path to model config YAML
            weights_path: Path to model weights
            device: CUDA device (e.g., "cuda:0", "cuda:1", "cpu")
            initial_subnet: Starting subnet configuration
        """
        print(f"[DetectionService] Initializing...")
        print(f"  Config: {config_path}")
        print(f"  Weights: {weights_path}")
        print(f"  Device: {device}")
        print(f"  Initial subnet: {initial_subnet}")

        # Import detectron2 components (lazy to avoid import errors if not needed)
        from detectron2.data import MetadataCatalog
        from mask2former.engine.defaults import DefaultPredictor

        # Load model configuration
        self._cfg = get_config(config_path, weights_path, device)
        self._predictor = DefaultPredictor(self._cfg)
        self._device = device

        # Get metadata for label mapping
        self._metadata = None
        if len(self._cfg.DATASETS.TRAIN):
            self._metadata = MetadataCatalog.get(self._cfg.DATASETS.TRAIN[0])
        elif len(self._cfg.DATASETS.TEST):
            self._metadata = MetadataCatalog.get(self._cfg.DATASETS.TEST[0])

        # Build label map
        self._label_map = build_label_map(self._metadata)
        print(f"  Label map: {len(self._label_map)} ADE20K -> Pylot mappings")

        # Set initial subnet (always use set_subnet, which handles ResNet gracefully)
        self._current_subnet = initial_subnet
        set_subnet(self._predictor.model, initial_subnet)

        print(f"[DetectionService] Ready on {device}")

    def Detect(self, request: detection_pb2.DetectRequest,
               context: grpc.ServicerContext) -> detection_pb2.DetectResponse:
        """
        Handle detection request.
        """
        total_start = time.perf_counter()

        try:
            # Deserialize image from bytes
            image = np.frombuffer(request.image_data, dtype=np.uint8)
            image = image.reshape((request.height, request.width, request.channels))

            # Handle subnet override for this request
            subnet_to_use = self._current_subnet
            if request.subnet and request.subnet != self._current_subnet:
                if request.subnet in MODEL_NETWORK_CONFIGS:
                    set_subnet(self._predictor.model, request.subnet)
                    subnet_to_use = request.subnet
                    # Note: we don't update self._current_subnet for per-request overrides

            # Set defaults for optional fields
            min_score = request.min_score if request.min_score > 0 else 0.3
            min_area = request.min_area if request.min_area > 0 else 200.0

            # Run inference
            inference_start = time.perf_counter()
            detections = run_inference(
                self._predictor,
                image,
                self._label_map,
                min_score=min_score,
                min_area=min_area,
                all_labels=request.all_labels,
                metadata=self._metadata
            )
            inference_time_ms = (time.perf_counter() - inference_start) * 1000

            # Restore subnet if we did a per-request override
            if request.subnet and request.subnet != self._current_subnet:
                set_subnet(self._predictor.model, self._current_subnet)

            # Build response
            response = detection_pb2.DetectResponse(
                success=True,
                inference_time_ms=inference_time_ms,
                subnet_used=subnet_to_use,
            )

            for det in detections:
                response.detections.append(detection_pb2.Detection(
                    x_min=det['bbox'][0],
                    y_min=det['bbox'][1],
                    x_max=det['bbox'][2],
                    y_max=det['bbox'][3],
                    label=det['label'],
                    score=det['score'],
                    class_index=det.get('class_idx', -1),
                ))

            response.total_time_ms = (time.perf_counter() - total_start) * 1000
            return response

        except Exception as e:
            return detection_pb2.DetectResponse(
                success=False,
                error=str(e),
                total_time_ms=(time.perf_counter() - total_start) * 1000,
            )

    def SetSubnet(self, request: detection_pb2.SetSubnetRequest,
                  context: grpc.ServicerContext) -> detection_pb2.SetSubnetResponse:
        """
        Switch the active subnet configuration.
        """
        try:
            subnet_name = request.subnet
            if subnet_name not in MODEL_NETWORK_CONFIGS:
                return detection_pb2.SetSubnetResponse(
                    success=False,
                    error=f"Unknown subnet: {subnet_name}. Available: {list(MODEL_NETWORK_CONFIGS.keys())}",
                    active_subnet=self._current_subnet,
                )

            set_subnet(self._predictor.model, subnet_name)
            self._current_subnet = subnet_name

            return detection_pb2.SetSubnetResponse(
                success=True,
                active_subnet=self._current_subnet,
            )

        except Exception as e:
            return detection_pb2.SetSubnetResponse(
                success=False,
                error=str(e),
                active_subnet=self._current_subnet,
            )

    def GetStatus(self, request: detection_pb2.StatusRequest,
                  context: grpc.ServicerContext) -> detection_pb2.StatusResponse:
        """
        Return current service status.
        """
        return detection_pb2.StatusResponse(
            ready=True,
            active_subnet=self._current_subnet,
            available_subnets=list(MODEL_NETWORK_CONFIGS.keys()),
            device=self._device,
            weights_path=self._cfg.MODEL.WEIGHTS,
        )


def serve(port: int, config_path: str, weights_path: str,
          device: str = "cuda:0", subnet: str = "max",
          max_workers: int = 4):
    """
    Start the gRPC server.

    Args:
        port: Port to listen on
        config_path: Path to model config
        weights_path: Path to model weights
        device: CUDA device
        subnet: Initial subnet configuration
        max_workers: Max thread pool workers
    """
    # Create servicer
    servicer = DetectionServicer(
        config_path=config_path,
        weights_path=weights_path,
        device=device,
        initial_subnet=subnet,
    )

    # Create server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    detection_pb2_grpc.add_DetectionServiceServicer_to_server(servicer, server)

    # Bind to port
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)

    print(f"\n[gRPC Server] Starting on port {port}")
    print(f"[gRPC Server] Listening on {listen_addr}")

    server.start()

    print("[gRPC Server] Ready for requests")
    print("[gRPC Server] Press Ctrl+C to stop\n")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[gRPC Server] Shutting down...")
        server.stop(grace=5)
        print("[gRPC Server] Stopped")


def main():
    parser = argparse.ArgumentParser(
        description='gRPC server for WS-Mask2Former detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start server with defaults
    python grpc_server.py

    # Start on specific port with middle subnet
    python grpc_server.py --port 50051 --subnet middle

    # Use specific GPU
    python grpc_server.py --device cuda:1
""")

    # Paths (with defaults relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, 'sem_ade_output_2', 'config.yaml')
    default_weights = os.path.join(script_dir, 'sem_ade_output_2', 'model_final.pth')

    parser.add_argument('--port', type=int, default=50051,
                        help='Port to listen on (default: 50051)')
    parser.add_argument('--config', default=default_config,
                        help='Path to config YAML')
    parser.add_argument('--weights', default=default_weights,
                        help='Path to model weights')
    parser.add_argument('--device', default='cuda:0',
                        help='Device (cuda:0, cuda:1, cpu)')
    parser.add_argument('--subnet', default='max',
                        choices=['min', 'middle', 'max'],
                        help='Initial subnet (default: max)')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Max thread pool workers (default: 4)')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.config):
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)
    if not os.path.exists(args.weights):
        print(f"ERROR: Weights not found: {args.weights}")
        sys.exit(1)

    serve(
        port=args.port,
        config_path=args.config,
        weights_path=args.weights,
        device=args.device,
        subnet=args.subnet,
        max_workers=args.max_workers,
    )


if __name__ == '__main__':
    main()
