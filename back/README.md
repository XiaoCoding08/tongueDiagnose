# 后端项目

## 目录结构

### common

- config.py: 配置文件，定义了使用的模型的路径，图像分类标签等

### entity

定义了相关的类

- Classify.py: 图像分类
- Detector.py: 目标检测
- MyResNet.py: 定义图像分类的网络结构

### model

存放相关的模型文件

### tmp

临时存放接受到的图像数据



## 项目启动

1. 运行命令 `pip install -r requirements.txt` 安装所需依赖
2. 运行 main.py 即可启动项目，项目默认在5000端口启动