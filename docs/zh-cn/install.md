# 安装指南

本文档提供了在不同操作系统上设置 speech2motion 开发环境的分步说明。

## 目录

- [Linux 环境设置](#linux-环境设置)
  - [Linux 前置条件](#linux-前置条件)
  - [Linux 步骤1：安装 Protocol Buffers](#linux-步骤1安装-protocol-buffers)
  - [Linux 步骤2：设置 Python](#linux-步骤2设置-python)
  - [Linux 步骤3：安装项目](#linux-步骤3安装项目)
  - [Linux 步骤4：验证安装](#linux-步骤4验证安装)
  - [Linux 环境激活](#linux-环境激活)
- [Windows 环境设置](#windows-环境设置)
  - [Windows 前置条件](#windows-前置条件)
  - [Windows 步骤1：安装 Protocol Buffers](#windows-步骤1安装-protocol-buffers)
  - [Windows 步骤2：设置 Python](#windows-步骤2设置-python)
  - [Windows 步骤3：安装项目](#windows-步骤3安装项目)
  - [Windows 步骤4：验证安装](#windows-步骤4验证安装)
  - [Windows 环境激活](#windows-环境激活)

## Linux 环境设置

### Linux 前置条件

在开始之前，请确保您满足以下系统要求：
- Ubuntu 20.04 或兼容的 Linux 发行版
- 用于下载软件包的网络连接

### Linux 步骤1：安装 Protocol Buffers

下载并安装用于pb文件编译的 protoc：

```bash
# 创建 protoc 目录
mkdir -p protoc
cd protoc

# 下载 protoc
curl -LjO https://github.com/protocolbuffers/protobuf/releases/download/v31.1/protoc-31.1-linux-x86_64.zip

# 解压并设置权限
unzip protoc-31.1-linux-x86_64.zip
rm -f protoc-31.1-linux-x86_64.zip
chmod +x bin/protoc

# 验证安装
bin/protoc --version

# 返回根目录
cd ..
```

### Linux 步骤2：设置 Python

您需要 Python 3.10 或更高版本来运行此项目。本文档提供了使用 conda 安装 Python 的方法作为参考。

**使用 Miniconda 安装 Python：**

```bash
# 下载 Miniconda 安装程序
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装 Miniconda
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda

# 清理安装程序
rm -f Miniconda3-latest-Linux-x86_64.sh

# 配置 conda 频道
conda config --add channels conda-forge
conda tos accept

# 使用 Python 3.10 创建 speech2motion 环境
conda create -n speech2motion python=3.10 -y

# 激活环境
conda activate speech2motion

```

### Linux 步骤3：安装项目

安装 speech2motion 包：

```bash
# 确保您在项目根目录
cd /path/to/speech2motion

# 激活 conda 环境
conda activate speech2motion

# 安装包
pip install .
```

### Linux 步骤4：验证安装

测试一切是否正常工作：

```bash
# 激活环境
conda activate speech2motion

# 检查是否可以导入 speech2motion.apis
python -c "import speech2motion.apis; print('speech2motion.apis imported successfully')"

# 检查主应用程序是否运行
python main.py --help
```

### Linux 环境激活

要使用 speech2motion 项目，请始终首先激活 conda 环境：

```bash
# 激活环境
conda activate speech2motion

# 您的终端现在应该显示 (speech2motion)
# 您现在可以运行 Python 脚本并使用 speech2motion 包
```


## Windows 环境设置

### Windows 前置条件

在开始之前，请确保您满足以下系统要求：
- Windows 10/11 或兼容的 Windows 发行版
- 用于下载软件包的网络连接

### Windows 步骤1：安装 Protocol Buffers

下载并安装用于协议缓冲区编译的 protoc：

1. **下载 protoc：**
   - 访问 [Protocol Buffers v31.1 发布页面](https://github.com/protocolbuffers/protobuf/releases/tag/v31.1)
   - 下载 Windows 版本：`protoc-31.1-win64.zip`

2. **解压文件：**
   - 在项目根目录中创建 `protoc` 文件夹
   - 将下载的 `protoc-31.1-win64.zip` 文件解压到 `protoc` 文件夹中
   - 确保可执行文件位于：`protoc\bin\protoc.exe`

3. **验证安装：**
   ```bash
   # 在项目目录中打开命令提示符
   protoc\bin\protoc.exe --version
   ```

### Windows 步骤2：设置 Python

您需要 Python 3.10 或更高版本来运行此项目。本文档提供了使用 conda 安装 Python 的方法作为参考。

**使用 Miniconda 安装 Python：**

1. **下载并安装 Miniconda：**
   - 访问 [Miniconda 安装指南](https://www.anaconda.com/docs/getting-started/miniconda/install)
   - 从 Anaconda 网站下载 Windows 安装程序
   - 按照官方安装说明安装 Miniconda
   - **重要**：在安装过程中，确保选中"将 Miniconda3 添加到我的 PATH 环境变量"或在 PATH 环境变量中手动添加 Miniconda3/Scripts 目录，以便从任何终端使用 conda 命令

2. **创建并激活环境：**
   ```bash
   # 使用 Python 3.10 创建 speech2motion 环境
   conda create -n speech2motion python=3.10 -y
   
   # 激活环境
   conda activate speech2motion
   ```

### Windows 步骤3：安装项目

安装 speech2motion 包：

```bash
# 确保您在项目根目录
cd /path/to/speech2motion

# 激活 conda 环境
conda activate speech2motion

# 临时将 protoc 添加到此会话的 PATH
set PATH=%PATH%;%CD%\protoc\bin

# 安装包
pip install .
```

### Windows 步骤4：验证安装

测试一切是否正常工作：

```bash
# 激活环境
conda activate speech2motion

# 检查是否可以导入 speech2motion.apis
python -c "import speech2motion.apis; print('speech2motion.apis imported successfully')"

# 检查主应用程序是否运行
python main.py --help
```

### Windows 环境激活

要使用 speech2motion 项目，请始终首先激活 conda 环境：

```bash
# 激活环境
conda activate speech2motion

# 您的终端现在应该显示 (speech2motion)
# 您现在可以运行 Python 脚本并使用 speech2motion 包
```



