#!/bin/bash

# install_cann.sh - 安装CANN的脚本
# 用法: ./install_cann.sh ARCH
# 参数: ARCH - CPU架构 (x86|arm)

show_help() {
    cat << EOF
Usage: $0 ARCH

Arguments:
    ARCH    CPU architecture (x86|arm)

Examples:
    $0 x86
    $0 arm
EOF
}

# 函数：检测操作系统类型
detect_os() {
    local os_name=""

    if [ -f /etc/os-release ]; then
        # 读取操作系统信息
        source /etc/os-release
        os_name=$(echo "$NAME" | tr '[:upper:]' '[:lower:]')
    fi

    echo "$os_name"
}

# 操作系统安装函数定义
# 函数命名格式: install_cann_{NPU_TYPE}_{ARCH}_{OS_TYPE}

# a2 NPU安装函数
install_cann_a2_x86_apt() {
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash

    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)

    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
}

install_cann_a2_x86_yum() {
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash

    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install
}

install_cann_a2_arm_apt() {
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
}

install_cann_a2_arm_yum() {
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
}

# a3 NPU安装函数
install_cann_a3_x86_apt() {
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
}

install_cann_a3_x86_yum() {
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-x86_64.run
    bash ./Ascend-cann_8.5.0_linux-x86_64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install
}

install_cann_a3_arm_apt() {
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo apt-get update
    sudo apt-get install -y gcc python3 python3-pip linux-headers-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
}

install_cann_a3_arm_yum() {
    groupadd HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo yum makecache
    sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
    #Ascend-cann为驱动、Toolkit合一包
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann_8.5.0_linux-aarch64.run
    bash ./Ascend-cann_8.5.0_linux-aarch64.run --install

    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
    bash ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
}

# 操作系统到包管理器的映射
declare -A OS_TO_PM=(
    [ubuntu]="apt"
    [debian]="apt"
    [velinux]="apt"
    [openeuler]="yum"
    [centos]="yum"
    [kylin]="yum"
    [bclinux]="yum"
    [uosv20]="yum"
    [antos]="yum"
    [alios]="yum"
    [ctyunos]="yum"
    [culinux]="yum"
    [tlinux]="yum"
    [mtos]="yum"
)

# 主函数
main() {
    # 检查是否提供了参数
    if [ $# -eq 0 ]; then
        echo "Error: No architecture specified"
        show_help
        return 1
    fi

    # 获取参数
    local ARCH="$1"

    # 验证参数
    if [ "$ARCH" != "x86" ] && [ "$ARCH" != "arm" ]; then
        echo "Error: Invalid architecture '$ARCH'. Must be 'x86' or 'arm'"
        return 1
    fi

    echo "========================================"
    echo "CANN Installation"
    echo "========================================"
    echo ""
    echo "Detected architecture: $ARCH"

    # 检测操作系统类型
    local OS_TYPE=$(detect_os)
    if [ -z "$OS_TYPE" ]; then
        echo "Error: Unable to detect operating system"
        return 1
    fi
    echo "Detected OS: $OS_TYPE"

    # 获取包管理器类型
    local PM_TYPE="${OS_TO_PM[$OS_TYPE]}"

    # 验证操作系统是否支持
    if [ -z "$PM_TYPE" ]; then
        echo "Error: Unsupported operating system: $OS_TYPE"
        echo "Supported operating systems:"
        printf "  APT based: "
        for os in "${!OS_TO_PM[@]}"; do
            if [ "${OS_TO_PM[$os]}" = "apt" ]; then
                printf "$os "
            fi
        done
        printf "\n  YUM based: "
        for os in "${!OS_TO_PM[@]}"; do
            if [ "${OS_TO_PM[$os]}" = "yum" ]; then
                printf "$os "
            fi
        done
        echo ""
        return 1
    fi

    echo "Package manager: $PM_TYPE"

    # 检测NPU硬件类型
    echo ""
    echo "Detecting NPU hardware type..."

    local npu_device_id=$(lspci -n -D | grep -o '19e5:d[0-9a-f]\{3\}' | head -n1 | cut -d: -f2)

    if [ -z "$npu_device_id" ]; then
        echo "Error: No NPU device found via lspci command."
        echo "Please ensure NPU device is properly installed and detected."
        echo "Command 'lspci -n -D | grep -o \"19e5:d[0-9a-f]\{3\}\"' returned empty result."
        return 1
    fi

    echo "Detected NPU device ID: $npu_device_id"

    # 转换为小写方便比较
    local npu_device_id_lower=$(echo "$npu_device_id" | tr '[:upper:]' '[:lower:]')
    local NPU_TYPE=""

    # 根据设备ID确定NPU类型（忽略大小写，完全匹配）
    case "$npu_device_id_lower" in
        d802)
            NPU_TYPE="a2"
            echo "NPU Type: Ascend A2 (d802)"
            ;;
        d803)
            NPU_TYPE="a3"
            echo "NPU Type: Ascend A3 (d803)"
            ;;
        *)
            echo "Error: Unsupported NPU device ID: $npu_device_id"
            echo "Supported device IDs: d802 (Ascend Atlas A2), d803 (Ascend Atlas A3)"
            echo "Your device ID: $npu_device_id"
            return 1
            ;;
    esac

    echo ""
    echo "System Configuration:"
    echo "  Architecture: $ARCH"
    echo "  OS: $OS_TYPE"
    echo "  Package Manager: $PM_TYPE"
    echo "  NPU Type: $NPU_TYPE"

    # 构建安装函数名
    local INSTALL_FUNC="install_cann_${NPU_TYPE}_${ARCH}_${PM_TYPE}"
    echo "Looking for installation function: $INSTALL_FUNC"

    # 检查安装函数是否存在
    if ! declare -f "$INSTALL_FUNC" > /dev/null; then
        echo "Error: Installation function '$INSTALL_FUNC' not found"
        echo ""
        echo "Available installation functions:"
        declare -f | grep "^install_cann_" | awk '{print $1}' | sed 's/()$//' | sort
        return 1
    fi

    echo ""
    echo "Starting CANN installation for NPU $NPU_TYPE on $ARCH $OS_TYPE ($PM_TYPE)..."

    # 执行对应的安装函数
    $INSTALL_FUNC

    # 检查安装结果
    if [ $? -eq 0 ]; then
        echo ""
        echo "CANN installation completed successfully!"
        echo "  OS: $OS_TYPE"
        echo "  Architecture: $ARCH"
        echo "  Package Manager: $PM_TYPE"
        echo "  NPU Type: $NPU_TYPE"
        return 0
    else
        echo ""
        echo "CANN installation failed!"
        echo "  OS: $OS_TYPE"
        echo "  Architecture: $ARCH"
        echo "  Package Manager: $PM_TYPE"
        echo "  NPU Type: $NPU_TYPE"
        return 1
    fi
}

# 调用主函数
main "$@"

# 根据主函数返回值退出
exit $?
