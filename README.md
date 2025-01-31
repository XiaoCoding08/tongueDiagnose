# 舌诊小程序



## 项目简介

这是一个基于微信小程序、flask框架、神经网络以及YOLO目标检测实现的舌诊应用程序，实现了用户拍摄舌头照片并上传至服务器进行体质诊断的功能。诊断结果包括用户的体质类型及推荐的养生饮品。



## 功能列表

1. 用户拍照功能
2. 照片上传至服务器
3. 显示诊断结果
4. 推荐相关饮品及价格



## 技术栈

### 前端

开发环境：微信开发者工具 Stable 1.06.2405020

本项目使用的技术栈包括：

- **小程序开发框架**: 微信小程序
- **前端框架**: WXML, WXSS, JavaScript
- **网络请求**: wx.request()



### 后端：

- **轻量级 Web 框架:** Flask
- **深度学习框架：** Pytorch 
- **目标检测:** YOLOv8 ultralytics



### 模型训练

- **深度学习框架:** Pytorch
- **图像分类网络框架:** ResNet34
- **目标检测:** YOLOv8  ultralytics



## 目录介绍以及项目启动

- 前端: front
- 后端: back
- 模型训练: TrainModel

各项目具体结构以及项目启动方法请查看对应目录下的README文件