#!/bin/bash
# WS-Mask2Former Inference Testing Script
# Encodes common testing commands with easy-to-use flags
#
# Usage: ./test_inference.sh <command> [options]
#
# Examples:
#   ./test_inference.sh infer -i /path/to/image.jpg -s max
#   ./test_inference.sh benchmark -i /path/to/image.jpg
#   ./test_inference.sh compare -i /path/to/image.jpg -o ./results

set -e

# Defaults
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/results"
SUBNET="max"
MIN_SCORE="0.1"
DOCKER_IMAGE="ws-mask2former"
IMAGE_PATH=""
LOG_FILE=""
ALL_LABELS=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
WS-Mask2Former Inference Testing Script

Usage: $0 <command> [options]

Commands:
  build       Build the Docker image
  labels      Print ADE20K label mapping
  infer       Run inference on a single image
  benchmark   Run latency benchmark across all subnets
  compare     Run all subnets on same image, save visualizations
  shell       Start interactive shell in container

Options:
  -i, --image PATH      Input image path (default: input.jpg in container)
  -o, --output-dir DIR  Output directory for results (default: ./results)
  -s, --subnet NAME     Subnet: min, middle, max (default: max)
  -t, --threshold NUM   Min detection score 0-1 (default: 0.1)
  -l, --log FILE        Log all output to file (overwrites each run)
  -a, --all-labels      Show all ADE20K labels (not just Pylot vehicle classes)
  -h, --help            Show this help

Examples:
  # Build Docker image
  $0 build

  # Run inference with max subnet on custom image
  $0 infer -i /path/to/image.jpg -s max

  # Run inference with lower threshold
  $0 infer -i /path/to/image.jpg -t 0.05

  # Benchmark all subnets
  $0 benchmark -i /path/to/image.jpg

  # Compare all subnets with visualizations saved to results/
  $0 compare -i /path/to/image.jpg -o ./results

  # Use default input.jpg from container
  $0 infer -s min
EOF
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

check_docker_image() {
    if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
        log_error "Docker image '$DOCKER_IMAGE' not found."
        log_info "Run: $0 build"
        exit 1
    fi
}

check_image_file() {
    if [ -n "$IMAGE_PATH" ] && [ ! -f "$IMAGE_PATH" ]; then
        log_error "Image file not found: $IMAGE_PATH"
        exit 1
    fi
}

ensure_output_dir() {
    mkdir -p "$OUTPUT_DIR"
}

# Log wrapper - if LOG_FILE is set, tee output to it
run_with_log() {
    if [ -n "$LOG_FILE" ]; then
        "$@" 2>&1 | tee "$LOG_FILE"
    else
        "$@"
    fi
}

# Get absolute path
get_abs_path() {
    if [ -n "$1" ]; then
        echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    fi
}

# Build docker run command with optional image mount
run_docker() {
    local extra_args=("$@")
    local cmd_args=()

    cmd_args+=(docker run --rm --gpus all)

    # If custom image provided, mount it
    if [ -n "$IMAGE_PATH" ]; then
        local abs_image=$(get_abs_path "$IMAGE_PATH")
        local image_dir=$(dirname "$abs_image")
        local image_name=$(basename "$abs_image")
        cmd_args+=(-v "$image_dir:/images:ro")
        cmd_args+=("${extra_args[@]}")
        cmd_args+=("$DOCKER_IMAGE" python inference_server.py)
        cmd_args+=(--image "/images/$image_name")
    else
        cmd_args+=("${extra_args[@]}")
        cmd_args+=("$DOCKER_IMAGE" python inference_server.py)
        cmd_args+=(--image "input.jpg")
    fi

    "${cmd_args[@]}"
}

# Parse command first
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

COMMAND="$1"
shift

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            IMAGE_PATH="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--subnet)
            SUBNET="$2"
            shift 2
            ;;
        -t|--threshold)
            MIN_SCORE="$2"
            shift 2
            ;;
        -l|--log)
            LOG_FILE="$2"
            shift 2
            ;;
        -a|--all-labels)
            ALL_LABELS="--all-labels"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute command
case "$COMMAND" in
    build)
        log_header "Building Docker Image"
        log_info "Image name: $DOCKER_IMAGE"
        cd "$SCRIPT_DIR"
        docker build -t "$DOCKER_IMAGE" .
        log_info "Build complete!"
        ;;

    labels)
        check_docker_image
        log_header "ADE20K Label Mapping"
        docker run --rm --gpus all "$DOCKER_IMAGE" python inference_server.py --print-labels
        ;;

    infer)
        check_docker_image
        check_image_file
        ensure_output_dir

        if [ -n "$IMAGE_PATH" ]; then
            IMAGE_PATH=$(get_abs_path "$IMAGE_PATH")
            IMAGE_DIR=$(dirname "$IMAGE_PATH")
            IMAGE_NAME=$(basename "$IMAGE_PATH")
            OUTPUT_NAME="${IMAGE_NAME%.*}_${SUBNET}.jpg"
        else
            IMAGE_DIR=""
            IMAGE_NAME="input.jpg"
            OUTPUT_NAME="input_${SUBNET}.jpg"
        fi

        log_header "Running Inference"
        log_info "Image: ${IMAGE_PATH:-input.jpg (built-in)}"
        log_info "Subnet: $SUBNET"
        log_info "Threshold: $MIN_SCORE"
        log_info "Output: $OUTPUT_DIR/$OUTPUT_NAME"

        if [ -n "$IMAGE_PATH" ]; then
            run_with_log docker run --rm --gpus all \
                -v "$IMAGE_DIR":/images:ro \
                -v "$OUTPUT_DIR":/output \
                "$DOCKER_IMAGE" python inference_server.py \
                --image "/images/$IMAGE_NAME" \
                --subnet "$SUBNET" \
                --min-score "$MIN_SCORE" \
                --output "/output/$OUTPUT_NAME" \
                $ALL_LABELS
        else
            run_with_log docker run --rm --gpus all \
                -v "$OUTPUT_DIR":/output \
                "$DOCKER_IMAGE" python inference_server.py \
                --image "input.jpg" \
                --subnet "$SUBNET" \
                --min-score "$MIN_SCORE" \
                --output "/output/$OUTPUT_NAME" \
                $ALL_LABELS
        fi

        log_info "Visualization saved: $OUTPUT_DIR/$OUTPUT_NAME"
        [ -n "$LOG_FILE" ] && log_info "Log saved: $LOG_FILE"
        ;;

    benchmark)
        check_docker_image
        check_image_file
        ensure_output_dir

        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        REPORT_FILE="$OUTPUT_DIR/benchmark_${TIMESTAMP}.txt"

        log_header "Latency Benchmark"
        log_info "Image: ${IMAGE_PATH:-input.jpg (built-in)}"
        log_info "Threshold: $MIN_SCORE"
        log_info "Report: $REPORT_FILE"
        echo ""

        if [ -n "$IMAGE_PATH" ]; then
            IMAGE_PATH=$(get_abs_path "$IMAGE_PATH")
            IMAGE_DIR=$(dirname "$IMAGE_PATH")
            IMAGE_NAME=$(basename "$IMAGE_PATH")

            docker run --rm --gpus all \
                -v "$IMAGE_DIR":/images:ro \
                "$DOCKER_IMAGE" python inference_server.py \
                --image "/images/$IMAGE_NAME" \
                --benchmark \
                --min-score "$MIN_SCORE" \
                $ALL_LABELS | tee "$REPORT_FILE" ${LOG_FILE:+| tee -a "$LOG_FILE"}
        else
            docker run --rm --gpus all \
                "$DOCKER_IMAGE" python inference_server.py \
                --image "input.jpg" \
                --benchmark \
                --min-score "$MIN_SCORE" \
                $ALL_LABELS | tee "$REPORT_FILE" ${LOG_FILE:+| tee -a "$LOG_FILE"}
        fi

        echo ""
        log_info "Report saved: $REPORT_FILE"
        [ -n "$LOG_FILE" ] && log_info "Log saved: $LOG_FILE"
        ;;

    compare)
        check_docker_image
        check_image_file
        ensure_output_dir

        if [ -n "$IMAGE_PATH" ]; then
            IMAGE_PATH=$(get_abs_path "$IMAGE_PATH")
            IMAGE_DIR=$(dirname "$IMAGE_PATH")
            IMAGE_NAME=$(basename "$IMAGE_PATH")
            BASE_NAME="${IMAGE_NAME%.*}"
        else
            IMAGE_DIR=""
            IMAGE_NAME="input.jpg"
            BASE_NAME="input"
        fi

        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        COMPARE_DIR="$OUTPUT_DIR/compare_${BASE_NAME}_${TIMESTAMP}"
        mkdir -p "$COMPARE_DIR"

        log_header "Subnet Comparison"
        log_info "Image: ${IMAGE_PATH:-input.jpg (built-in)}"
        log_info "Threshold: $MIN_SCORE"
        log_info "Output dir: $COMPARE_DIR/"

        # Create comparison report
        REPORT_FILE="$COMPARE_DIR/comparison_report.txt"
        CSV_FILE="$COMPARE_DIR/results.csv"

        cat > "$REPORT_FILE" << EOF
=== Subnet Comparison Report ===
Image: ${IMAGE_PATH:-input.jpg}
Timestamp: $(date)
Threshold: $MIN_SCORE

EOF

        # CSV header
        echo "subnet,latency_ms,num_detections" > "$CSV_FILE"

        for subnet in min middle max; do
            echo ""
            log_info "Processing subnet: $subnet"
            OUTPUT_NAME="${BASE_NAME}_${subnet}.jpg"
            JSON_NAME="${BASE_NAME}_${subnet}.json"

            echo "--- Subnet: $subnet ---" >> "$REPORT_FILE"

            # Run inference - capture output separately for display and JSON extraction
            TEMP_OUTPUT="$COMPARE_DIR/.temp_output_$subnet"

            if [ -n "$IMAGE_PATH" ]; then
                docker run --rm --gpus all \
                    -v "$IMAGE_DIR":/images:ro \
                    -v "$COMPARE_DIR":/output \
                    "$DOCKER_IMAGE" python inference_server.py \
                    --image "/images/$IMAGE_NAME" \
                    --subnet "$subnet" \
                    --min-score "$MIN_SCORE" \
                    --output "/output/$OUTPUT_NAME" \
                    --json $ALL_LABELS 2>&1 | tee "$TEMP_OUTPUT"
            else
                docker run --rm --gpus all \
                    -v "$COMPARE_DIR":/output \
                    "$DOCKER_IMAGE" python inference_server.py \
                    --image "input.jpg" \
                    --subnet "$subnet" \
                    --min-score "$MIN_SCORE" \
                    --output "/output/$OUTPUT_NAME" \
                    --json $ALL_LABELS 2>&1 | tee "$TEMP_OUTPUT"
            fi

            # Extract just the JSON part (starts with '{', ends with '}')
            # This filters out PyTorch warnings
            sed -n '/^{/,/^}/p' "$TEMP_OUTPUT" > "$COMPARE_DIR/$JSON_NAME"
            rm -f "$TEMP_OUTPUT"

            # Extract metrics for CSV (parse clean JSON)
            if command -v python3 &> /dev/null && [ -s "$COMPARE_DIR/$JSON_NAME" ]; then
                latency=$(python3 -c "import json; d=json.load(open('$COMPARE_DIR/$JSON_NAME')); print(d.get('latency_ms', 'N/A'))" 2>/dev/null || echo "N/A")
                num_det=$(python3 -c "import json; d=json.load(open('$COMPARE_DIR/$JSON_NAME')); print(d.get('num_detections', len(d.get('detections', []))))" 2>/dev/null || echo "N/A")
                echo "$subnet,$latency,$num_det" >> "$CSV_FILE"
            else
                echo "$subnet,N/A,N/A" >> "$CSV_FILE"
            fi

            cat "$COMPARE_DIR/$JSON_NAME" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
        done

        echo ""
        log_header "Comparison Complete"
        echo ""
        echo "Generated artifacts:"
        echo "  Visualizations:"
        for f in "$COMPARE_DIR"/*.jpg; do
            [ -f "$f" ] && echo "    - $f"
        done
        echo "  JSON results:"
        for f in "$COMPARE_DIR"/*.json; do
            [ -f "$f" ] && echo "    - $f"
        done
        echo "  Report: $REPORT_FILE"
        echo "  CSV: $CSV_FILE"
        echo ""

        # Show CSV contents
        if [ -f "$CSV_FILE" ]; then
            log_info "Results summary:"
            column -t -s',' "$CSV_FILE" 2>/dev/null || cat "$CSV_FILE"
        fi
        ;;

    shell)
        check_docker_image
        log_header "Interactive Shell"
        docker run --rm -it --gpus all \
            -v "$SCRIPT_DIR":/workspace \
            "$DOCKER_IMAGE" \
            /bin/bash
        ;;

    -h|--help|help)
        usage
        exit 0
        ;;

    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
