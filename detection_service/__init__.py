# gRPC service definitions for wsobjdet
# Generated from detection.proto

from .detection_pb2 import (
    DetectRequest,
    DetectResponse,
    Detection,
    SetSubnetRequest,
    SetSubnetResponse,
    StatusRequest,
    StatusResponse,
)
from .detection_pb2_grpc import (
    DetectionServiceStub,
    DetectionServiceServicer,
    add_DetectionServiceServicer_to_server,
)

__all__ = [
    'DetectRequest',
    'DetectResponse',
    'Detection',
    'SetSubnetRequest',
    'SetSubnetResponse',
    'StatusRequest',
    'StatusResponse',
    'DetectionServiceStub',
    'DetectionServiceServicer',
    'add_DetectionServiceServicer_to_server',
]
