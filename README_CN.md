# Speech2Motion

> **中文文档** | [English Documentation](README.md)

## 目录

- [项目概述](#项目概述)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [文档](#文档)
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
   - **百度网盘**：访问 [动作数据库下载链接](https://pan.baidu.com/s/1YCisRewRQQdYT-GzCZxu-w?pwd=wwqm)
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
  dlp3d/speech2motion:latest
```

**Windows：**
```bash
# 拉取并运行预构建镜像
docker run -it -p 18084:18084 -v .\data:/workspace/speech2motion/data dlp3d/speech2motion:latest
```

**命令说明：**
- `-p 18084:18084`：将容器的18084端口映射到主机的18084端口
- `-v $(pwd)/data:/workspace/speech2motion/data`（Linux/macOS）：将本地 `data` 目录挂载到容器的数据目录
- `-v .\data:/workspace/speech2motion/data`（Windows）：将本地 `data` 目录挂载到容器的数据目录
- `dlp3d/speech2motion:latest`：使用预构建的公共镜像

**前提条件：**
- 确保项目根目录中有 `data` 目录
- 确保已安装并运行 Docker

## 文档

有关安装、API 使用、配置和开发的详细信息，请访问我们的完整文档：

📖 **[完整文档](https://dlp3d.readthedocs.io/zh-cn/latest/_subrepos/speech2motion/overview.html)**

文档包括：
- **安装指南**：分步环境设置和依赖项安装
- **API 文档**：详细的 API 规范和使用示例
- **配置说明**：本地和生产环境配置选项
- **开发指南**：项目结构、测试和贡献指南

## 许可证

本项目采用 MIT 许可证。详情请参见 [LICENSE](LICENSE) 文件。

MIT 许可证是一个宽松的开源许可证，允许您：
- 将软件用于任何目的
- 修改和分发软件
- 将软件包含在专有应用程序中
- 销售软件

唯一的要求是在软件的任何副本或重要部分中包含原始版权声明和许可证文本。

---
