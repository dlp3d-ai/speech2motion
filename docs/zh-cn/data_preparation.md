# 数据准备

要使用 Speech2Motion，您需要下载离线动作数据库并设置所需的目录结构。

## 下载动作数据库

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

## 目录结构说明

- `data/motion_files`：用于存储二进制动作文件的文件夹
- `data/restpose_npz/`：用于存储NPZ格式RestPose数据的文件夹
- `data/motion_database.db`：包含动作数据库的SQLite文件
- `data` 目录将挂载到Docker容器的 `/workspace/speech2motion/data`

