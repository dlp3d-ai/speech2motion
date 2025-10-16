# Speech2Motion

> **中文文档** | [English Documentation](../README.md)

## 目录

- [项目概述](#项目概述)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [环境配置](#环境配置)
- [API 文档](#api-文档)
- [配置说明](#配置说明)
- [开发指南](#开发指南)
- [许可证](#许可证)

## 项目概述

Speech2Motion 是一个实时流式系统，能够将语音输入转换为同步的3D角色动画。系统基于语音内容、关键词和时间进行智能动作匹配，为交互式应用提供自然且富有表现力的角色动画。

### 核心特性

- **实时流式处理**：支持低延迟的流式语音转动作转换
- **多版本API**：提供V1、V2和V3三个版本的API，具有不同的功能特性
- **智能匹配**：针对动作和语音文本内容的高级关键词匹配
- **记忆管理**：用户会话记忆，避免重复动画
- **灵活数据源**：支持多种数据后端（SQLite、MySQL、MinIO、本地文件系统）
- **动作融合**：不同动作序列之间的平滑过渡
- **角色支持**：多角色支持，可自定义RestPose
- **可扩展架构**：模块化设计，支持可插拔的筛选器和读取器

### 系统架构

系统由以下几个关键组件组成：

- **流式API**：处理实时语音输入和动作生成
- **动作数据库**：包含动作元数据和二进制文件的SQLite/MySQL数据库
- **筛选管道**：用于动作选择的多阶段筛选系统
- **时间线管理**：基于帧的动作序列时间线
- **记忆系统**：用户会话管理，跟踪最近见过的动作
- **文本处理**：基于Jieba的文本分割，用于关键词提取
- **动作合并**：插值和混合，实现平滑过渡

## 数据准备

要使用 Speech2Motion，您需要下载离线动作数据库并设置所需的目录结构。

### 下载动作数据库

1. **下载动作数据库：**
   - **Google Drive**：访问 [动作数据库下载链接](https://drive.google.com/file/d/112pnjuIuNqADS-fAT6RUIAVPtb3VlWlq/view?usp=drive_link)
   - **百度网盘**：访问 [动作数据库下载链接](https://pan.baidu.com/s/1YJSuLaoDKKV7JuE0Ws89zA)（提取码：`g64i`）
   - 根据您的网络情况选择合适的下载方式，下载压缩的动作数据库文件

2. **解压并组织数据：**
   - 将下载的文件解压到项目根目录
   - 确保创建以下目录结构：

```
├─configs
├─data
│  ├─motion_files
│  │  └─motion
│  ├─restpose_npz
│  └─motion_database.db
├─docs
├─speech2motion
└─tools
```

### 目录结构说明

- `data/motion_files`：用于存储二进制动作文件的文件夹
- `data/restpose_npz/`：用于存储NPZ格式RestPose数据的文件夹
- `data/motion_database.db`：包含动作数据库的SQLite文件
- `data` 目录将挂载到Docker容器的 `/workspace/speech2motion/data`

## 快速开始

### 使用 Docker

使用预构建的 Docker 镜像是开始使用 Speech2Motion 最简单的方式：

**Linux/macOS：**
```bash
# 拉取并运行预构建镜像
docker run -it \
  -p 18084:18084 \
  -v $(pwd)/data:/workspace/speech2motion/data \
  dockersenseyang/dlp3d_speech2motion:latest
```

**Windows：**
```cmd
# 拉取并运行预构建镜像
docker run -it -p 18084:18084 -v .\data:/workspace/speech2motion/data dockersenseyang/dlp3d_speech2motion:latest
```

**命令说明：**
- `-p 18084:18084`：将容器的18084端口映射到主机的18084端口
- `-v $(pwd)/data:/workspace/speech2motion/data`（Linux/macOS）：将本地 `data` 目录挂载到容器的数据目录
- `-v .\data:/workspace/speech2motion/data`（Windows）：将本地 `data` 目录挂载到容器的数据目录
- `dockersenseyang/dlp3d_speech2motion:latest`：使用预构建的公共镜像

**前提条件：**
- 确保项目根目录中有 `data` 目录
- 确保已安装并运行 Docker

**备选方案：从源码构建**

如果您希望从源码构建镜像：

**Linux/macOS：**
```bash
# 构建 Docker 镜像
docker build -t speech2motion:local .

# 运行容器
docker run -it \
  -p 18084:18084 \
  -v $(pwd)/data:/workspace/speech2motion/data \
  speech2motion:local
```

**Windows：**
```cmd
# 构建 Docker 镜像
docker build -t speech2motion:local .

# 运行容器
docker run -it -p 18084:18084 -v .\data:/workspace/speech2motion/data speech2motion:local
```

## 环境配置

对于本地开发和部署，请遵循详细的安装指南：

📖 **[完整安装指南](install.md)**

安装指南提供以下步骤说明：
- 设置 Python 3.10+ 环境
- 安装 Protocol Buffers 编译器
- 配置开发环境
- 安装项目依赖

### 本地开发

按照安装指南完成环境设置后，您可以在本地启动服务：

```bash
# 激活 conda 环境
conda activate speech2motion

# 启动服务
python main.py
```

## API 文档

### 流式 API

系统提供三个版本的流式 API，具有不同的功能和部署策略：

- **V1 API**：基础流式动作生成接口，具有基本动作关键词匹配功能。该版本目前已弃用，不再维护。

- **V2 API**：基于V1基础构建的增强流式接口，具有以下特性：
  - 针对情绪和关系等额外标注的高级检索能力
  - 增强对不同下游应用的支持
  - 改进的插值记忆管理

- **V3 API**：采用完全不同的双时间线同步检索策略的革命性流式接口：
  - 显著提高关键词动作触发率
  - 高级混合能力和改进的过渡效果
  - 双时间线架构，增强动作同步

**部署**：V2 和 V3 API 同时部署，可通过不同的 FastAPI 服务端点访问，允许应用程序根据特定需求选择最合适的版本。

### 请求/响应格式

所有 API 都使用 Protocol Buffers 进行高效序列化。系统支持：

- **分块处理**：语音输入分块处理，实现实时响应
- **动作时间线**：基于帧的时间线管理，实现精确的动作序列
- **关键词匹配**：动作关键词和语音关键词的智能选择
- **记忆集成**：用户会话记忆，避免重复动画

## 配置说明

### 本地配置

对于本地开发，系统使用 `configs/local.py`，配置以下内容：

- **SQLite 数据库**：本地动作数据库（`data/motion_database.db`）
- **文件系统读取器**：基于本地文件的动作和RestPose读取器

### 生产配置

对于生产部署，使用 `configs/diamond.py`，支持：

- **MySQL 数据库**：生产级数据库后端
- **MinIO 存储**：动作文件的对象存储

## 开发指南

### 项目结构

```
speech2motion/
├── apis/              # 流式 API 实现
├── cache/             # 缓存系统
├── data_structures/   # 核心数据模型
├── filters/           # 动作筛选管道
├── io/                # 数据 I/O 模块
├── merge/             # 动作合并和混合
├── retrieve/          # 动作检索系统
├── service/           # FastAPI 服务器
├── text_segmentation/ # 文本处理
├── utils/             # 工具函数
└── variety/           # 动作多样性管理
```

### 测试

项目包含全面的测试：

```bash
# 运行所有测试
pytest tests --log-cli-level=ERROR

# 运行特定测试类别
pytest tests/filters/  # 筛选器测试
pytest tests/merge/    # 动作合并测试
```

### 代码质量

项目通过以下方式保持高代码质量：

- **代码检查**：使用 Ruff 进行代码风格和质量检查
- **类型提示**：完整的类型注解支持
- **CI/CD**：自动化测试和部署管道

## 许可证

本项目采用 MIT 许可证。详情请参见 [LICENSE](../LICENSE) 文件。

MIT 许可证是一个宽松的开源许可证，允许您：
- 将软件用于任何目的
- 修改和分发软件
- 将软件包含在专有应用程序中
- 销售软件

唯一的要求是在软件的任何副本或重要部分中包含原始版权声明和许可证文本。

---
