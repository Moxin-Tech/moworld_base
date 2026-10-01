
# 安全声明

## 系统安全加固

1. 建议用户在系统中配置开启ASLR（级别2），又称**全随机地址空间布局随机化**，可参考以下方式进行配置：

    ```bash
    echo 2 > /proc/sys/kernel/randomize_va_space
    ```

## 运行用户建议

出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户使用MindSpeed MM。

## 文件权限控制

1. 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
2. 建议用户对个人数据、商业资产、源文件、训练过程中保存的各类文件等敏感内容做好权限管控。涉及场景如MindSpeed MM安装目录权限管控、多用户使用共享数据集权限管控，管控权限可参考表1进行设置。
3. MindSpeed MM在数据预处理中会生成训练数据，在训练过程会生成权重文件，文件权限默认640，用户可根据实际需求对生成文件权限进行进阶管控。

**表1 文件（夹）各场景权限管控推荐最大值**

| 类型          | Linux权限参考最大值 |
| --------------- | --------------------|
| 用户主目录                          |    750（rwxr-x---）                |
| 程序文件（含脚本文件、库文件等）      |    550（r-xr-x---）                |
| 程序文件目录                        |    550（r-xr-x---）                |
| 配置文件                            |    640（rw-r-----）                |
| 配置文件目录                        |    750（rwxr-x---）                |
| 日志文件（记录完毕或者已经归档）      |    440（r--r-----）                |
| 日志文件（正在记录）                 |    640（rw-r-----）                |
| 日志文件记录                        |    750（rwxr-x---）                |
| Debug文件                          |    640（rw-r-----）                |
| Debug文件目录                      |    750 (rwxr-x---)                 |
| 临时文件目录                       |     750（rwxr-x---）                |
| 维护升级文件目录                    |    770（rwxrwx---）                |
| 业务数据文件                       |     640（rw-r-----）                |
| 业务数据文件目录                   |     750（rwxr-x---）                |
| 密钥组件、私钥、证书、密文文件目录   |     700（rwx------）                |
| 密钥组件、私钥、证书、加密密文      |     600（rw-------）                |
| 加解密接口、加解密脚本             |     500（r-x------）                |

## 数据安全声明

1. 风险概述：
   MindSpeed MM 模型框架执行模型的加载与保存操作。需要特别注意的是，其底层实现可能使用 [Python pickle](https://docs.python.org/3/library/pickle.html)模块进行部分文件的序列化/反序列化操作，该模块存在固有的安全风险。

2. 核心风险场景：
   当使用 PyTorch 提供的 torch.load()方法加载模型文件时，一个关键的安全风险点在于设置 weights_only=False。在此设置下：

   特定框架实现： Megatron-LM 框架的原生代码调用、MindSpeed MM提供的权重转换脚本（将 Megatron 格式转换为 Hugging Face 格式）中，会显式地将 weights_only=False。这意味着这些加载操作继承了 pickle模块的潜在危险，允许执行任意代码。
   攻击面： 攻击者可能通过构造恶意的模型文件，利用 pickle的反序列化漏洞实现远程代码执行 (RCE)。

3. 严重漏洞警示 (CVE-2025-32434)

   即使将 weights_only设置为 True，用户仍面临严重风险，特别是使用 PyTorch 版本 <= 2.5.1 时：

   攻击者可以利用旧版的 .tar格式模型文件构造恶意模型。这种构造可以绕过 weights_only=True的安全检查机制。成功利用可触发 RCE。用户必须参考此漏洞说明：CVE-2025-32434。

4. 关键安全防护措施

   鉴于上述高风险，强烈建议：

   来源可信： 仅加载来自官方发布渠道或高度可信仓库的模型文件。
   完整性验证： 下载后，务必使用密码学哈希值（如 SHA-256）校验模型文件的完整性和来源真实性。
   环境隔离： 在隔离的环境（例如容器或沙箱）中运行模型加载代码，并严格限制该环境的系统访问权限（沙箱逃逸是另一个方向的安全问题）。
   安全工具： 利用专用的安全工具（如针对 Pickle 的扫描器）对模型文件进行检测，识别潜在的恶意序列化对象。
   PyTorch 版本升级： 避免使用 PyTorch <= 2.5.1 版本。立即升级到已修复 CVE-2025-32434 漏洞的更高版本。

5. 参考链接

   torch.load()文档：<https://pytorch.org/docs/main/generated/torch.load.html#torch.load>(包含 weights_only参数说明及其风险)

   PyTorch 分布式通信说明：<https://pytorch.org/docs/main/distributed.html#collective-functions>

## 运行安全声明

1. 建议用户结合运行资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。
2. MindSpeed MM内部用到了PyTorch和torch_npu,可能会因为版本不匹配导致运行错误，具体可参考PyTorch及torch_npu[安全声明](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.md)。
3. MindSpeed MM的依赖库transformers和datasets在使用from_pretrained方法时，存在配置trust_remote_code=True的情况。此设置会直接执行从远程仓库下载的代码，可能包含恶意逻辑或后门程序，导致系统面临代码注入攻击等安全威胁。用户需要确保自己下载的模型和数据的安全性。

## MoWorld 预训练入口

本分支提供 [pretrain_sora.py](./pretrain_sora.py) 预训练入口及 [MoWorld 数据准备和训练脚本](./examples/moworld/)。环境、数据与权重准备方法见[使用指南](../USAGE.md)。

## 通信安全加固

[通信安全加固说明](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA)

## 通信矩阵

[通信矩阵说明](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E7%9F%A9%E9%98%B5)

## 特殊场景

| 场景                                                                             | 使用方法                                                                                                        | 端口           | 可能的风险                                                   |
|--------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------| -------------- | ------------------------------------------------------------ |
| 使用MindSpeed MM进行训练任务时，在Megatron后端场景下每次初始化模型并行组时，默认新增（3 \* NPU数量）个随机端口，开启多个分布式优化器时再额外新增（分布式优化器数量 \* NPU数量）个随机端口，同时配置1个master-port端口（该端口与torch_npu的master-port端口一致） | MindSpeed MM 调用 Megatron 原生函数 `mpu.initialize_model_parallel` 来初始化模型并行组，并通过使用 PyTorch 分布式训练相关的 API 来启动任意任务。 | [1024,65520]内 | 网络配置错误可能引发端口冲突或连接问题，影响训练效率。       |

### 公网地址声明

代码涉及公网地址参考 [public_address_statement.md](./docs/zh/public_address_statement.md)
