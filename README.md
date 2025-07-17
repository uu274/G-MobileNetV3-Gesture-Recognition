# Enhanced Physician Gesture Recognition via G-MobileNetv3 with Group-Mix Attention Mechanism
**Published in The Visual Computer**

## 🔍 Project Overview  
This project is based on the MobileNetV3 framework and introduces a series of innovative structural improvements for intraoperative physician gesture recognition. Our work focuses on the following three key aspects:

### 🔧 Group-Mix Attention Mechanism  
A novel Group-Mix Attention (GMA) module is integrated into the backbone to replace the traditional SE attention mechanism. This module models both local and distant feature dependencies, significantly improving the model’s ability to focus on key gesture features in complex surgical environments, especially under varying lighting conditions and strong background interference.

### 🔄 Cross-Stage Residual Connectivity  
Inspired by the design of DenseNet, cross-stage residual connections are introduced between non-adjacent bottleneck blocks. This design improves feature reuse and information flow, effectively preserving initial gesture features while maintaining the lightweight structure of the model, making it well-suited for real-time applications.

### ⚙️ PReLU Activation Function  
To address issues like gradient vanishing and neuron inactivation associated with ReLU, we replace it with the learnable PReLU activation function. This enhances the model’s adaptability to diverse gesture distributions and improves convergence and generalization performance on surgical gesture datasets.

---

## 🧩 Module Descriptions

- **Group-Mix Attention Module**  
  Enhances the model’s ability to capture dependencies between local and distant regions in the image, allowing it to focus more accurately on key gesture features under complex conditions.

- **Cross-Stage Residual Connectivity Module**  
  Builds skip connections between different stages of the network to effectively preserve early-stage features and enhance feature flow, thereby improving model representation and training stability.

- **PReLU Activation Function**  
  Introduces a learnable parameter for negative inputs to alleviate the dead neuron issue in ReLU and enhances the model’s ability to adapt to complex gesture distributions.

- **G-MobileNetV3 Overall Structure**  
  While maintaining lightweight and high efficiency, the G-MobileNetV3 network integrates GMA attention, cross-stage connectivity, and the PReLU activation function to comprehensively enhance recognition accuracy and robustness in surgical scenarios.

---

## ✅ Summary  
With the above enhancements, this project achieves accurate, robust, and real-time intraoperative physician gesture recognition. Please refer to our detailed documentation for full API usage instructions and model architecture information to ensure transparency and reproducibility of experiments.

---


