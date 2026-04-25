# CIFAR10-RESNET18# 不同优化算法对 ResNet18 在 CIFAR-10 数据集上训练效果的分析

《人工智能中的数学方法》课程项目  

## 项目介绍
本项目基于 PyTorch 框架，使用 ResNet18 作为骨干网络，在 CIFAR-10 数据集上对比 **SGD、SGD+Momentum、Adam、AdamW** 四种优化算法的训练表现，分析不同优化器在收敛速度、验证集精度、过拟合程度及泛化能力上的差异，为深度学习任务中优化器选择提供实验依据。

## 实验环境
- Python 3.x
- PyTorch
- CIFAR-10 数据集


## 主要内容
1. 四种优化算法原理与公式
2. ResNet18 网络结构修改（适配 CIFAR-10 32×32 输入）
3. 数据增强与训练配置
4. 训练损失、验证准确率对比
5. 实验结果与结论

## 优化器对比
- SGD：基础随机梯度下降
- SGD+Momentum：带动量，收敛更稳定
- Adam：自适应学习率，收敛快
- AdamW：解耦权重衰减，泛化更好

## 文件结构
├── train.py # 主训练脚本
├── model.py # 改造后 ResNet18 模型
├── README.md # 项目说明文档
└── report.md # 完整实验报告

## 运行方式
1. 安装依赖
```bash
pip install torch torchvision
2.启动训练
python train.py

##实验结论

自适应类优化器（Adam、AdamW）收敛速度远快于传统 SGD 系列；
带动量的 SGD 训练更加平稳，后期准确率上限较高；
AdamW 有效抑制过拟合，综合泛化能力最优；
传统 SGD 对学习率设置高度敏感，调参成本更高。

##团队成员
黄子懿、郭成、高子程
