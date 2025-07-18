# G-MobileNetV3-Gesture-Recognition

## Project Description

This repository contains **G-MobileNetV3**, a lightweight yet robust model for physician gesture recognition in complex surgical environments. The network integrates three key innovations:

* **Group-Mix Attention (GMA)** – reinforces attention to critical spatial–channel features.
* **Cross-Stage Residual Connection (CSRC)** – improves feature propagation and gradient flow.
* **PReLU Activation** – adapts more effectively to diverse gesture patterns under strong illumination or background clutter.

For full technical details, please refer to our paper:

> **Enhanced Physician Gesture Recognition via G-MobileNetv3 with Group-Mix Attention Mechanism**, submitted to *The Visual Computer*, 2025.

## Dataset

We employ the **Surgery-Gesture Dataset**, specifically collected for gesture recognition in real surgical scenes.

Download link:

📂 **[Surgery-Gesture Dataset on Google Drive](https://drive.google.com/drive/folders/101s5aNbuW0mgAzPqKJ-PqivyTrKvGG63)**

## Project Structure

```
G-MobileNetV3-Gesture-Recognition/
├── model/
│   └── G-MobileNetV3.py          # Network architecture
│   └── group_mix_attention.py    # Group-Mix Attention 
├── tool/
│   ├── predict.py                # Inference script
│   ├── time.py                   # Runtime benchmarking
│   └── train.py                  # Training script (contains all hyper-parameters)
├── requirements.txt              # Python dependencies
├── setup.py                      # Optional: build script if you use Cython
└── LICENSE
```

## Environment Setup

* **Python 3.8** recommended.
* Install required packages:

```bash
pip install -r requirements.txt
```

## Training

All training hyper-parameters (learning rate, batch size, number of epochs, dataset paths, etc.) are **defined directly inside `tool/train.py`**.
To start training:

```bash
python tool/train.py
```

If you need to adjust any parameter, simply edit the corresponding variables at the top of `tool/train.py`.

## Inference

To run gesture recognition on an image with a trained model:

```bash
python tool/predict.py --model_path your_model.pth --image_path path_to_image.jpg
```

## Module Overview

* **`G-MobileNetV3.py`** – full network definition, embedding GMA, CSRC, and PReLU.
* Lightweight **GMA ** attention mechanisms enhance feature sensitivity.
* **Cross-stage residual connections** preserve early-stage features and improve deep-layer fusion.

## Citation

If this project contributes to your research, please cite:

```bibtex
@article{wang2025gmobile,
  title={Enhanced Physician Gesture Recognition via G-MobileNetv3 with Group-Mix Attention Mechanism},
  author={Wenjie Wang and Xu Yang and Xiaohua Wang and Huajian Song},
  journal={The Visual Computer},
  year={2025}
}
```

## License

Released under the **MIT License** (see `LICENSE`).

## Contact

**Corresponding author:** Prof. Huajian Song
📧 [songhuajian01@163.com](mailto:songhuajian01@163.com)



