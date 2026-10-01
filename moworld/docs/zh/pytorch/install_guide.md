# MoWorld 环境准备

MoWorld 的安装命令见[主 README](../../../../README.md#installation)，模型与预训练配置见[使用指南](../../../../USAGE.md)。当前实现位于 `moworld/`，无需另行克隆 MindSpeed-MM。

## 平台环境

使用具备昇腾 NPU 的 Linux 环境，并根据硬件型号准备匹配的驱动、固件、CANN、PyTorch 和 `torch_npu`。当前包的依赖要求见 [pyproject.toml](../../../pyproject.toml)：Python >= 3.10、PyTorch 2.7.1。软件版本之间的配套关系以所用平台的安装文档为准。

- [硬件与操作系统兼容性](https://www.hiascend.com/hardware/compatibility)
- [驱动与固件](https://hiascend.com/hardware/firmware-drivers/community)
- [CANN 软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html)
- [Ascend Extension for PyTorch 安装](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)

运行前加载实际安装位置的 CANN 环境脚本；启动脚本支持通过 `ASCEND_ENV` 指定路径。

## 安装项目

按照[使用指南的 Installation](../../../../USAGE.md#installation) 安装仓库中固定版本的 MindSpeed 子模块、Megatron-LM 和 `moworld` 包。内部导入名为 `mindspeed_mm`。

## 容器部署

容器需要能够访问 NPU 设备及配套的驱动、运行库，并挂载模型、输入和输出目录。设备映射与环境配置请遵循所用昇腾镜像的说明。进入容器后，按[训练说明](../../../../USAGE.md#training) 设置路径并运行 MoWorld 预训练脚本。

安全说明见 [SECURITYNOTE.md](../../../SECURITYNOTE.md)。预训练仍需在目标昇腾环境中完成验证。
