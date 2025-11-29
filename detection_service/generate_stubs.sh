#!/bin/bash
# Generate Python gRPC stubs from detection.proto
# Run this from the grpc/ directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Generating Python gRPC stubs..."

python3 -m grpc_tools.protoc \
    --proto_path=. \
    --python_out=. \
    --grpc_python_out=. \
    detection.proto

echo "Generated:"
echo "  - detection_pb2.py (message classes)"
echo "  - detection_pb2_grpc.py (service stubs)"

# Fix the import in grpc file (relative import issue)
sed -i 's/import detection_pb2/from . import detection_pb2/' detection_pb2_grpc.py 2>/dev/null || \
sed -i '' 's/import detection_pb2/from . import detection_pb2/' detection_pb2_grpc.py 2>/dev/null || \
echo "Note: Could not fix relative import - may need manual fix"

echo "Done!"
