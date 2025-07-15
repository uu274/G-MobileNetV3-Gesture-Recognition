# G-MobileNetV3-Gesture-Recognition

## 项目介绍

本项目提供了基于G-MobileNetV3网络改进的医生手势识别模型，通过融合Group-Mix Attention（GMA）模块、跨阶段残差连接（Cross-stage Residual Connection）以及PReLU激活函数，提升了模型在复杂手术环境下（如强光干扰、背景复杂性）的手势识别性能。

本项目代码为论文提供了直接实现方案，使用时请引用本项目相关论文：

> ** G-MobileNetv3: Physician gesture recognition combined with Group-Mix Attention, submitted to *The Visual Computer*, 2024.**

## 数据集下载

本项目使用的数据集已公开，可通过以下Google Drive链接下载：

* [Surgery-Gesture Dataset](https://drive.google.com/drive/folders/101s5aNbuW0mgAzPqKJ-PqivyTrKvGG63)

## 项目结构

```
├── model
│   └── g_mobilenetv3.py
├── tool
│   ├── tool.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
├── setup.py
└── config.yaml
```

## 环境配置

推荐使用Python 3.8环境。

安装依赖：

```bash
pip install -r requirements.txt
```

## 模型训练

在配置文件 `config.yaml` 中设置参数（如数据路径、学习率、批次大小等），然后运行：

```bash
python tool/train.py
```

## 模型推理

使用训练好的模型进行预测，执行以下命令：

```bash
python tool/predict.py --model_path your_model.pth --image_path path_to_image.jpg
```

## 模块说明

* **Attention模块（attention.py）**: 包含GMA和SIMAM注意力机制。
* **Cross-stage模块（cross\_stage.py）**: 提供跨阶段残差连接，增强网络初始特征的保留。
* **主模型（g\_mobilenetv3.py）**: 完整的G-MobileNetV3结构定义。

## 引用方式

如果您使用了本项目的代码或数据，请务必引用本项目的相关论文，以支持我们的研究工作：

```bibtex
@article{wang2024gmobile,
  title={G-MobileNetv3: Physician gesture recognition combined with Group-Mix Attention},
  author={Wenjie Wang, Xu Yang, Xiaohua Wang, Huajian Song},
  journal={The Visual Computer},
  year={2024},
}
```

## License

本项目采用MIT许可证，详见LICENSE文件。

## 联系我们

如有任何问题，欢迎通过邮箱联系：

* 通讯作者: [wangwenjie@xpu.edu.cn](mailto:wangwenjie@xpu.edu.cn)

---

感谢您关注并使用我们的研究成果！

