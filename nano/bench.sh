#!/usr/bin/env bash
# Build and time TensorRT engines on the Jetson Nano, FP32 and FP16, for any ONNX model.
# Adapted from github.com/aviadarn/offroad-lidar-degradation (bench/jetson_trt.sh).
#
# Run ON the Nano:   bash bench.sh ~/jdp/policy.onnx policy
#                    bash bench.sh ~/jdp/yolov5s_640.onnx yolo640
#
# Why the numbers are trustworthy:
#  - GPU clocks pinned first (an idle Nano sits at 76.8 MHz vs a 921.6 MHz ceiling).
#  - One timing cache shared across builds, so FP32/FP16 pick tactics on equal terms.
#  - No INT8: the Nano is Maxwell (SM 5.3), no DP4A. TensorRT accepts --int8, falls back,
#    and emits the FP16 engine; measured in offroad-lidar-degradation, 1.00x vs FP16.
set -u
ONNX=${1:?usage: bench.sh model.onnx name}
NAME=${2:-$(basename "$ONNX" .onnx)}
DIR=$(dirname "$ONNX")
TRTEXEC=/usr/src/tensorrt/bin/trtexec
CACHE=$DIR/timing.cache
ITERS=${ITERS:-200}

sudo -n nvpmodel -m 0 2>/dev/null
sudo -n jetson_clocks 2>/dev/null
echo "power mode: $(sudo -n nvpmodel -q 2>/dev/null | head -1)"
echo "GPU clock: $(cat /sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq 2>/dev/null) Hz"

for prec in fp32 fp16; do
  log=$DIR/${NAME}_${prec}.log
  extra=""; [ "$prec" = fp16 ] && extra="--fp16"
  echo "=== $NAME $prec ==="
  $TRTEXEC --onnx="$ONNX" --workspace=1024 --iterations=$ITERS --avgRuns=20 --percentile=99 \
           --timingCacheFile="$CACHE" --saveEngine="$DIR/${NAME}_${prec}.engine" \
           $extra > "$log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "  FAILED (rc=$rc)"; tail -3 "$log"; continue; fi
  grep -E "GPU Compute Time: |Throughput: " "$log" | sed 's/^.*\] //; s/^/  /'
  echo "  engine: $(ls -l "$DIR/${NAME}_${prec}.engine" | awk '{printf "%.1f MB", $5/1048576}')"
done
