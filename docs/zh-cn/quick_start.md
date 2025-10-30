# 快速开始

## 使用 Docker

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
```bash
# 构建 Docker 镜像
docker build -t speech2motion:local .

# 运行容器
docker run -it -p 18084:18084 -v .\data:/workspace/speech2motion/data speech2motion:local
```

