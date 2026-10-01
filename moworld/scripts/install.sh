#!/bin/bash

INSTALL_PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$INSTALL_PROJECT_DIR" || exit 1

# show help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -t, --torchversion VERSION   PyTorch version to install (default: 2.7.1)
    -m, --msid COMMIT_ID    MindSpeed commit ID [required]
    -y, --yes               Auto confirm all reinstallations
    -n, --no                Auto skip all reinstallations
    -mt, --megatron         Install Megatron-LM
    -ic, --install-cann     Install CANN (Compute Architecture for Neural Networks)
    -h, --help              Display this help message and exit

Examples:
    # Install everything including CANN
    bash $0 --torchversion 2.7.1 --msid 93c45456c7044bacddebc5072316c01006c938f9 --install-cann

    # Install without CANN
    bash $0 --torchversion 2.7.1 --msid 93c45456c7044bacddebc5072316c01006c938f9

    # Auto confirm all reinstallations
    bash $0 --torchversion 2.6.0 --msid 93c45456c7044bacddebc5072316c01006c938f9 --yes

    # Auto skip all reinstallations
    bash $0 --msid abcdef1234567890 --no

    # Interactive mode (default)
    bash $0 --torchversion 2.7.1 --msid abcdef1234567890
EOF
}

# Default values
TORCH_VERSION="2.7.1"
MINDSPEED_COMMIT_ID=""
AUTO_CONFIRM=""  # Auto confirm mode: "", "yes", "no"
INSTALL_MEGATRON=false  # Whether to install Megatron-LM
INSTALL_CANN=false  # Whether to install CANN

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--torchversion) TORCH_VERSION="$2"; shift 2 ;;
        -m|--msid) MINDSPEED_COMMIT_ID="$2"; shift 2 ;;
        -y|--yes) AUTO_CONFIRM="yes"; shift ;;
        -n|--no) AUTO_CONFIRM="no"; shift ;;
        -mt|--megatron) INSTALL_MEGATRON=true; shift ;;
        -ic|--install-cann) INSTALL_CANN=true; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown parameter: $1"; show_help; exit 1 ;;
    esac
done

# Check required parameters
if [ -z "$MINDSPEED_COMMIT_ID" ]; then
    echo "Error: MindSpeed commit ID parameter is required"
    show_help
    exit 1
fi

echo "========================================"
echo "Installation Configuration"
echo "========================================"
echo "PyTorch Version: $TORCH_VERSION"
echo "MindSpeed Commit ID: $MINDSPEED_COMMIT_ID"
echo "Auto Confirm Mode: ${AUTO_CONFIRM:-"interactive"}"
echo "Install Megatron-LM: $INSTALL_MEGATRON"
echo "Install CANN: $INSTALL_CANN"
echo "========================================"
echo ""

detect_device() {
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo "gpu"
        return 0
    fi

    if command -v npu-smi &> /dev/null && npu-smi info &> /dev/null; then
        echo "npu"
        return 0
    fi

    echo "Error: No GPU or NPU detected. This script requires GPU or NPU for installation."
    echo "Supported devices: NVIDIA GPU (nvidia-smi) or NPU (npu-smi)"
    exit 1
}

# Function to detect CUDA version
detect_cuda_version() {
    if command -v nvcc &> /dev/null; then
        nvcc --version | grep "release" | awk '{print $6}' | cut -c2-
    elif [ -f /usr/local/cuda/version.txt ]; then
        cat /usr/local/cuda/version.txt | awk '{print $3}'
    elif [ -f /usr/local/cuda/version.json ]; then
        grep -o '"cuda_version":[^,]*' /usr/local/cuda/version.json | cut -d'"' -f4
    else
        echo ""
    fi
}

get_cuda_index_url() {
    local cuda_version=$1

    # Select corresponding index URL based on CUDA version
    case $cuda_version in
        12.4|12.5|12.6|12.*)
            echo "https://download.pytorch.org/whl/cu124"
            ;;
        12.1|12.2|12.3)
            echo "https://download.pytorch.org/whl/cu121"
            ;;
        12.0)
            echo "https://download.pytorch.org/whl/cu120"
            ;;
        11.8)
            echo "https://download.pytorch.org/whl/cu118"
            ;;
        11.7)
            echo "https://download.pytorch.org/whl/cu117"
            ;;
        11.6)
            echo "https://download.pytorch.org/whl/cu116"
            ;;
        10.2)
            echo "https://download.pytorch.org/whl/cu102"
            ;;
        *)
            echo "https://download.pytorch.org/whl/cu121"
            ;;
    esac
}

# Function to detect CPU architecture
detect_architecture() {
    local arch
    arch=$(uname -m)

    case $arch in
        x86_64|X86_64|amd64|AMD64)
            echo "x86"
            ;;
        aarch64|AARCH64|arm64|ARM64)
            echo "arm"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Auto-detect CPU architecture
ARCH=$(detect_architecture)

# If auto-detection fails, ask user to input manually
if [ -z "$ARCH" ]; then
    echo "Unable to auto-detect CPU architecture."
    echo "Detected CPU architecture: $(uname -m)"
    echo "Please manually input CPU architecture (x86 or arm):"
    read -r user_arch

    # Convert to lowercase for comparison
    user_arch_lower=$(echo "$user_arch" | tr '[:upper:]' '[:lower:]')

    if [ "$user_arch_lower" = "x86" ]; then
        ARCH="x86"
    elif [ "$user_arch_lower" = "arm" ]; then
        ARCH="arm"
    else
        echo "Error: Unsupported architecture '$user_arch'"
        echo "Only x86 and arm architectures are supported."
        echo "Detected CPU architecture: $(uname -m)"
        exit 1
    fi
fi

echo "Detected CPU architecture: $ARCH"

DEVICE_TYPE=$(detect_device)
echo "Detected device type: $DEVICE_TYPE"

# If GPU device is detected, detect CUDA version
if [ "$DEVICE_TYPE" = "gpu" ]; then
    CUDA_VERSION=$(detect_cuda_version)
    echo "Detected CUDA version: $CUDA_VERSION"

    # Get corresponding PyTorch index URL
    CUDA_INDEX_URL=$(get_cuda_index_url "$CUDA_VERSION")
    echo "PyTorch index URL for CUDA $CUDA_VERSION: $CUDA_INDEX_URL"
fi

get_target_torch_version() {
    local target_version="$TORCH_VERSION"
    if [ "$DEVICE_TYPE" = "gpu" ]; then
        echo "GPU device, target version set to: $target_version (will use CUDA version)" >&2
    elif [ "$DEVICE_TYPE" = "npu" ]; then
        echo "NPU device, target version set to: $target_version" >&2
    fi
    echo "$target_version"
}

TARGET_TORCH_VERSION=$(get_target_torch_version)
echo "Final target Torch version: $TARGET_TORCH_VERSION"
echo ""

CANN_INSTALL_SCRIPT="scripts/install_cann.sh"

# Only execute CANN installation when --install-cann parameter is specified
if [ "$INSTALL_CANN" = true ]; then
    # Check if install_cann.sh exists
    if [ ! -f "$CANN_INSTALL_SCRIPT" ]; then
        echo "Error: $CANN_INSTALL_SCRIPT not found in current directory"
        echo "Please ensure install_cann.sh exists in $(pwd)"
        exit 1
    fi

    # Check if install_cann.sh has execute permission
    if [ ! -x "$CANN_INSTALL_SCRIPT" ]; then
        echo "Setting execute permission for install_cann.sh..."
        chmod +x "$CANN_INSTALL_SCRIPT"
    fi

    # Call install_cann.sh and pass ARCH parameter
    echo "Calling install_cann.sh with architecture: $ARCH"
    # Execute CANN installation script
    if ! "$CANN_INSTALL_SCRIPT" "$ARCH"; then
        cann_exit_code=$?
        echo "Error: CANN installation failed with exit code: $cann_exit_code"
        echo "Aborting installation due to CANN installation failure."
        exit 1
    fi

    # Validate CANN installation
    echo "Validating CANN installation..."

    # Change the ascend-toolkit path to the actual installation path.
    if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        # Verify if acl module can be imported
        if python3 -c "import acl; print(acl.get_soc_name())" 2>/dev/null | grep -qi ascend; then
            echo "CANN installation successful"
        else
            echo "Error: CANN validation failed - unable to import acl module"
            exit 1
        fi
    else
        echo "Error: CANN installation failed - ascend-toolkit not found"
        exit 1
    fi
else
    echo "Skipping CANN Installation ..."
    echo "  (Use --install-cann flag to install CANN)"
fi

# Function to check if package is installed
is_package_installed() {
    local package_name=$1
    pip3 show "$package_name" &>/dev/null
    return $?
}

# Function to install with retry
install_with_retry() {
    local package_name=$1
    local install_cmd=$2
    local max_retries=3
    local retry_count=0

    echo "Installing $package_name..."

    while [ $retry_count -lt $max_retries ]; do
        echo "Attempt $((retry_count + 1)) of $max_retries..."

        if $install_cmd; then
            # Check if installation was successful
            if is_package_installed "$package_name"; then
                echo "$package_name installed successfully!"
                return 0
            else
                echo "Installation command succeeded but package not found, retrying..."
            fi
        else
            echo "Installation command failed, retrying..."
        fi

        retry_count=$((retry_count + 1))

        # Wait 3 seconds before retry (except on last attempt)
        if [ $retry_count -lt $max_retries ]; then
            echo "Waiting 3 seconds before retry..."
            sleep 3
        fi
    done

    echo "Error: Failed to install $package_name after $max_retries attempts"
    return 1
}

# Function to get torch version
get_torch_version() {
    if is_package_installed "torch"; then
        pip3 show torch | grep "^Version:" | awk '{print $2}'
    else
        echo ""
    fi
}

# Version validation function
check_existing_versions() {
    local install_torch=true
    local install_torch_npu=true
    local message=""

    echo "Auto confirm mode: ${AUTO_CONFIRM:-"interactive (no auto confirm)"}"

    local target_torch_version="$TARGET_TORCH_VERSION"

    # Check if torch is already installed
    if is_package_installed "torch"; then
        local current_torch_version
        current_torch_version=$(get_torch_version)

        if [ "$current_torch_version" != "$target_torch_version" ]; then
            message+="\n=== PyTorch Version Mismatch ===\n"
            message+="Currently installed torch version: $current_torch_version\n"
            message+="Target version: $target_torch_version\n"
            message+="y: Reinstall PyTorch to target version\n"
            message+="n: Skip PyTorch installation, continue with other components\n"

            echo "Version check results:"
            echo -e "$message"

            # Process auto confirm
            if [ "$AUTO_CONFIRM" = "yes" ]; then
                echo "Auto confirming: Will reinstall PyTorch (--yes flag detected)"
                install_torch=true
            elif [ "$AUTO_CONFIRM" = "no" ]; then
                echo "Auto skipping: Will skip PyTorch installation (--no flag detected)"
                install_torch=false
            else
                # Interactive mode
                while true; do
                    echo "torch version mismatch detected. Reinstall PyTorch? (y/n)"
                    read -r user_input

                    case $user_input in
                        [Yy]* )
                            echo "Will reinstall PyTorch..."
                            install_torch=true
                            break
                            ;;
                        [Nn]* )
                            echo "Will skip PyTorch installation..."
                            install_torch=false
                            break
                            ;;
                        * )
                            echo "Invalid input. Please enter y or n"
                            ;;
                    esac
                done
            fi
        else
            echo "Current torch version matches target version: $target_torch_version"
            install_torch=false
        fi
    else
        echo "torch not detected, will install new version: $target_torch_version"
    fi

    # Determine whether to check torch_npu based on device type
    if [ "$DEVICE_TYPE" != "npu" ]; then
        echo "NPU environment not detected, skipping torch_npu installation"
        install_torch_npu=false

        # Return values
        echo "install_torch=$install_torch" > /tmp/install_flags
        echo "install_torch_npu=$install_torch_npu" >> /tmp/install_flags
        return 0
    fi

    # NPU environment detected, continue processing torch_npu
    echo "NPU environment detected, will check torch_npu version"
    install_torch_npu=true

    # Check if torch_npu is already installed
    if ! is_package_installed "torch_npu"; then
        echo "torch_npu not detected, will install new version: $TORCH_VERSION"

        # Return values
        echo "install_torch=$install_torch" > /tmp/install_flags
        echo "install_torch_npu=$install_torch_npu" >> /tmp/install_flags
        return 0
    fi

    # Already installed torch_npu, check if version matches
    local current_torch_npu_version
    current_torch_npu_version=$(pip3 show torch_npu | grep "^Version:" | awk '{print $2}')

    if [ "$current_torch_npu_version" = "$TORCH_VERSION" ]; then
        echo "Current torch_npu version matches target version: $TORCH_VERSION"
        install_torch_npu=false

        # Return values
        echo "install_torch=$install_torch" > /tmp/install_flags
        echo "install_torch_npu=$install_torch_npu" >> /tmp/install_flags
        return 0
    fi

    # Version mismatch, need user confirmation
    echo -e "\n=== torch_npu Version Mismatch ==="
    echo "Currently installed torch_npu version: $current_torch_npu_version"
    echo "Target version: $TORCH_VERSION"

    # Process auto confirm logic
    if [ "$AUTO_CONFIRM" = "yes" ]; then
        echo "Auto confirming: Will reinstall torch_npu (--yes flag detected)"
        install_torch_npu=true
    elif [ "$AUTO_CONFIRM" = "no" ]; then
        echo "Auto skipping: Will skip torch_npu installation (--no flag detected)"
        install_torch_npu=false
    else
        # Interactive mode
        while true; do
            echo "Reinstall torch_npu to match PyTorch version? (y/n)"
            read -r user_input

            case $user_input in
                [Yy]* )
                    echo "Will reinstall torch_npu..."
                    install_torch_npu=true
                    break
                    ;;
                [Nn]* )
                    echo "Will skip torch_npu installation..."
                    install_torch_npu=false
                    break
                    ;;
                * )
                    echo "Invalid input. Please enter y or n"
                    ;;
            esac
        done
    fi

    # Return values
    echo "install_torch=$install_torch" > /tmp/install_flags
    echo "install_torch_npu=$install_torch_npu" >> /tmp/install_flags
    return 0
}

# Execute version check
if ! check_existing_versions; then
    exit 1
fi

# Read installation flags
source /tmp/install_flags
rm -f /tmp/install_flags

echo ""
echo "Installation plan:"
echo "- Install torch: $install_torch"
echo "- Install torch_npu: $install_torch_npu"
echo "- Device type: $DEVICE_TYPE"
echo "- Target torch version: $TARGET_TORCH_VERSION"
echo ""

# Install PyTorch components
echo "Starting PyTorch components installation..."

# Install PyTorch if needed
if [ "$install_torch" = true ]; then
    echo "Installing PyTorch $TARGET_TORCH_VERSION..."
    if [ "$ARCH" = "x86" ] && [ "$DEVICE_TYPE" = "gpu" ]; then
        echo "Installing x86 GPU version of PyTorch $TARGET_TORCH_VERSION..."
        echo "Using CUDA index URL: $CUDA_INDEX_URL"
        pip3 install "torch==$TARGET_TORCH_VERSION" "torchvision" "torchaudio" --index-url "$CUDA_INDEX_URL"
    elif [ "$ARCH" = "x86" ] && [ "$DEVICE_TYPE" = "npu" ]; then
        echo "Installing x86 NPU version of PyTorch $TARGET_TORCH_VERSION..."
        pip3 install "torch" "torchvision" "torchaudio" --index-url "https://download.pytorch.org/whl/cpu"
    elif [ "$ARCH" = "arm" ] && [ "$DEVICE_TYPE" = "npu" ]; then
        echo "Installing ARM NPU version of PyTorch $TARGET_TORCH_VERSION..."
        pip3 install "torch==$TARGET_TORCH_VERSION" "torchvision" "torchaudio"
    else
        echo "Error: Unsupported combination: architecture='$ARCH', device='$DEVICE_TYPE'"
        echo "Supported combinations:"
        echo "  - x86 + gpu"
        echo "  - x86 + npu"
        echo "  - arm + npu"
        show_help
        exit 1
    fi

    if is_package_installed "torch"; then
        installed_torch=$(get_torch_version)
        echo "torch version: $installed_torch successfully installed!"
    else
        echo "[ERROR] Installation failed! Reason: torch installation error."
    fi
else
    current_torch_version=$(get_torch_version)
    echo "Using existing torch version: $current_torch_version"
    echo "Installing torchvision and torchaudio compatible with torch $current_torch_version..."

    if [ "$ARCH" = "x86" ] && [ "$DEVICE_TYPE" = "gpu" ]; then
        echo "Installing x86 GPU compatible packages..."
        if [ -n "$CUDA_INDEX_URL" ]; then
            pip3 install "torch==$current_torch_version" "torchvision" "torchaudio" --index-url "$CUDA_INDEX_URL"
        else
            pip3 install "torch==$current_torch_version" "torchvision" "torchaudio" --index-url "https://download.pytorch.org/whl/cu121"
        fi
    elif [ "$ARCH" = "x86" ] && [ "$DEVICE_TYPE" = "npu" ]; then
        echo "Installing x86 NPU compatible packages..."
        pip3 install "torch==$current_torch_version" "torchvision" "torchaudio"
    elif [ "$ARCH" = "arm" ] && [ "$DEVICE_TYPE" = "npu" ]; then
        echo "Installing ARM NPU compatible packages..."
        pip3 install "torch==$current_torch_version" "torchvision" "torchaudio"
    fi
fi

# Only install torch_npu when needed
if [ "$install_torch_npu" = true ]; then
    echo "Installing torch_npu $TORCH_VERSION..."
    pip3 install torch-npu=="$TORCH_VERSION"

    if is_package_installed "torch_npu"; then
        installed_npu=$(pip3 show torch_npu | grep "^Version:" | awk '{print $2}')
        echo "torch_npu version: $installed_npu successfully installed!"
    else
        echo "[ERROR] Installation failed! Reason: torch_npu installation error."
    fi
elif is_package_installed "torch_npu"; then
    current_torch_npu_version=$(pip3 show torch_npu | grep "^Version:" | awk '{print $2}')
    echo "Using existing torch_npu version: $current_torch_npu_version"
fi

# Install megatron (only when -mt or --megatron parameter is specified)
if [ "$INSTALL_MEGATRON" = true ]; then
    echo "[INFO] Installing Megatron-LM..."
    cd ..
    if [ ! -d "Megatron-LM" ]; then
        git clone https://github.com/NVIDIA/Megatron-LM.git
    fi
    cd Megatron-LM
    git checkout core_v0.12.1
    if [ ! -d "megatron" ]; then
        echo "[ERROR] Installation failed! Reason: Megatron-LM installation error."
        exit 1
    fi

    cp -r megatron "$INSTALL_PROJECT_DIR/"
    cd "$INSTALL_PROJECT_DIR/" || exit 1

    echo "Megatron-LM successfully installed!"
else
    echo "[INFO] Skipping Megatron-LM installation (not specified with -mt or --megatron)"
fi

# Install mindspeed with retry mechanism
echo "[INFO] Installing MindSpeed with commit ID: $MINDSPEED_COMMIT_ID"
if [ ! -d "MindSpeed" ]; then
    git clone https://gitcode.com/Ascend/MindSpeed.git
fi
cd MindSpeed
git checkout "$MINDSPEED_COMMIT_ID"

# Install MindSpeed with retry mechanism
if ! install_with_retry "mindspeed" "pip3 install -e ."; then
    echo "[ERROR] Installation failed! Reason: MindSpeed installation failed after multiple attempts."
    exit 1
fi

echo "[INFO] MindSpeed with commit ID: $MINDSPEED_COMMIT_ID successfully installed!"

cd ..

# Create directories
echo "[INFO] Creating necessary directories..."
mkdir -p logs data ckpt

# Install mindspeed-mm dependency library with retry mechanism
echo "[INFO] Installing mindspeed-mm dependency library..."
if ! install_with_retry "mindspeed-mm" "pip3 install -e ."; then
    echo "[ERROR] Installation failed! Reason: mindspeed-mm installation failed after multiple attempts."
    exit 1
fi

packages=("mindspeed-mm" "mindspeed")
all_found=true

for pkg in "${packages[@]}"; do
    if ! pip3 list 2>/dev/null | grep -q "^${pkg} "; then
        all_found=false
        break
    fi
done

if $all_found; then
    echo "[INFO] mindspeed mm successfully installed!"
else
    echo "[ERROR] Installation failed! Reason: mindspeed-mm or mindspeed install failed."
fi
