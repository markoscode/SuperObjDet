#!/bin/bash
# Helper script to run the gRPC detection server
# Usage:
#   ./run_grpc_server.sh           # Run server on default port 50051
#   ./run_grpc_server.sh build     # Rebuild Docker image first
#   ./run_grpc_server.sh test      # Run server and test client in parallel
#   ./run_grpc_server.sh shell     # Open shell in container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="ws-mask2former"
CONTAINER_NAME="wsobjdet-grpc"
GRPC_PORT="${GRPC_PORT:-50051}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  (none)    Run gRPC server"
    echo "  build     Rebuild Docker image"
    echo "  test      Run server and test client"
    echo "  shell     Open shell in container"
    echo "  stop      Stop running container"
    echo "  logs      Show container logs"
    echo ""
    echo "Environment:"
    echo "  GRPC_PORT  Port to expose (default: 50051)"
}

build_image() {
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker build -t "$IMAGE_NAME" .
    echo -e "${GREEN}Build complete!${NC}"
}

run_server() {
    # Stop any existing container
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    echo -e "${YELLOW}Starting gRPC server on port $GRPC_PORT...${NC}"
    docker run --gpus all \
        --name "$CONTAINER_NAME" \
        -p "$GRPC_PORT:50051" \
        --rm \
        "$IMAGE_NAME" \
        python grpc_server.py --port 50051

    # This blocks - server runs in foreground
}

run_server_background() {
    # Stop any existing container
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    echo -e "${YELLOW}Starting gRPC server in background on port $GRPC_PORT...${NC}"
    docker run --gpus all \
        --name "$CONTAINER_NAME" \
        -p "$GRPC_PORT:50051" \
        -d \
        "$IMAGE_NAME" \
        python grpc_server.py --port 50051

    echo -e "${GREEN}Server started. Use './run_grpc_server.sh logs' to view output.${NC}"
}

run_test() {
    # First ensure server is running
    if ! docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
        echo -e "${YELLOW}Server not running. Starting in background...${NC}"
        run_server_background
        echo "Waiting for server to initialize (10s)..."
        sleep 10
    fi

    echo -e "${YELLOW}Running test client...${NC}"
    docker exec "$CONTAINER_NAME" python grpc_client_test.py --server localhost:50051

    echo ""
    echo -e "${YELLOW}Running benchmark...${NC}"
    docker exec "$CONTAINER_NAME" python grpc_client_test.py --server localhost:50051 --benchmark --num-runs 5
}

open_shell() {
    echo -e "${YELLOW}Opening shell in container...${NC}"
    docker run --gpus all \
        --name "${CONTAINER_NAME}-shell" \
        -p "$GRPC_PORT:50051" \
        --rm \
        -it \
        "$IMAGE_NAME" \
        /bin/bash
}

stop_server() {
    echo -e "${YELLOW}Stopping server...${NC}"
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    echo -e "${GREEN}Stopped.${NC}"
}

show_logs() {
    docker logs -f "$CONTAINER_NAME"
}

# Main
case "${1:-}" in
    help|--help|-h)
        show_help
        ;;
    build)
        build_image
        ;;
    test)
        run_test
        ;;
    shell)
        open_shell
        ;;
    stop)
        stop_server
        ;;
    logs)
        show_logs
        ;;
    "")
        run_server
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
