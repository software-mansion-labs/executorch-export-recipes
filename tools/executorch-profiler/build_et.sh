# ------------------------
# Python environment check
# ------------------------

if [ -z "$VIRTUAL_ENV" ]; then
  echo "No active Python environment detected. Please activate a virtual environment."
  exit 1
else
  echo "Active Python environment detected: $VIRTUAL_ENV"
fi

# ----------------------
# Argument Parsing
# ----------------------

BUILD_COREML="OFF"

for arg in "$@"
do
    case $arg in
        --coreml)
        BUILD_COREML="ON"
        shift
        ;;
    esac
done

# ----------------------
# CMake arguments setup
# ----------------------

# ET_EVENT_TRACER_ENABLED is essential to enable generating the ETDump file
CMAKE_ARGS=" \
-DEXECUTORCH_BUILD_XNNPACK=ON \
-DEXECUTORCH_BUILD_COREML=${BUILD_COREML} \
-DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
-DEXECUTORCH_BUILD_KERNELS_LLM=ON \
-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
-DEXECUTORCH_BUILD_DEVTOOLS=ON \
-DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
-DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
-Dprotobuf_BUILD_TESTS=OFF \
-DCMAKE_BUILD_TYPE=Release"

# -----------------
# Extra setup/build
# -----------------

if [ "$BUILD_COREML" == "ON" ]; then
    echo "Building CoreML runner..."
  
    # Install CoreML requirements
    CMAKE_ARGS="${CMAKE_ARGS}" ./backends/apple/coreml/scripts/install_requirements.sh

    # Build CoreML runner
    CMAKE_ARGS="${CMAKE_ARGS}" ./examples/apple/coreml/scripts/build_executor_runner.sh
    echo "CoreML build enabled."
fi

# ------------------------
# Installation script call
# ------------------------

CMAKE_ARGS="${CMAKE_ARGS}" ./install_executorch.sh

# ---------------------------
# Python package installation
# ---------------------------

# Add missing Python packages for profiling script
pip install colorama
