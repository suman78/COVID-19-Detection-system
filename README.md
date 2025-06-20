# 🦠 COVID-19 Detection System Using MobileNetV2

An intelligent, fast, and lightweight Deep Learning-based solution for detecting COVID-19 infection from chest X-ray images using **MobileNetV2**.

## 🚀 Project Overview

The COVID-19 pandemic has emphasized the need for quick and reliable diagnostic tools. This project utilizes **MobileNetV2**, a lightweight and efficient convolutional neural network, to classify chest X-ray images into three categories:
- **COVID-19 Positive**
- **Viral Pneumonia**
- **Normal (Healthy)**

The system provides high accuracy with minimal computational resources, making it suitable for real-time applications and deployment on mobile or edge devices.

---

## 🧠 Core Technologies

| Tool / Library       | Purpose                                      |
|----------------------|----------------------------------------------|
| Python               | Core programming language                    |
| TensorFlow & Keras   | Deep learning model development              |
| MobileNetV2          | Pre-trained CNN used for feature extraction  |
| OpenCV               | Image processing                             |
| Matplotlib & Seaborn | Data visualization                           |
| Scikit-learn         | Model evaluation (accuracy, confusion matrix)|

---

## 📂 Dataset

- **Source:** Public COVID-19 X-ray datasets from [Kaggle](https://www.kaggle.com/)
- **Classes:** `COVID`, `Pneumonia`, `Normal`
- **Preprocessing Steps:**
  - Image resizing to 224x224
  - Normalization
  - Data augmentation (rotation, zoom, flip)

---

## 🔍 Model Architecture

- Based on **MobileNetV2** – a lightweight CNN optimized for mobile and embedded vision applications.
- The final layers are customized using:
  - Global Average Pooling
  - Fully Connected Dense layers
  - Softmax for 3-class classification

---

## 📊 Results

- **Accuracy:** 97% on test set
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam (Adaptive learning rate)
- Confusion Matrix and Performance Metrics:
  
  ![Confusion Matrix](results/confusion_matrix.png)
  ![Accuracy Graph](results/accuracy_graph.png)

---

## 💻 How to Run

### 🔧 Installation

```bash
git clone https://github.com/yourusername/covid19-detection-mobilenetv2.git
cd covid19-detection-mobilenetv2
pip install -r requirements.txt
