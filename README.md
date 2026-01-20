<<<<<<< HEAD
# 👁️ SmartVision AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20PyTorch-orange?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge&logo=streamlit&logoColor=white)

**SmartVision AI** is a **real-world Computer Vision application** that performs **multi-object detection and image classification** using **YOLOv8** and **CNN-based deep learning models**.

The project is built for **practical usage, demos, and deployment**, moving beyond simple experimentation to a production-ready workflow.

---

## 🚀 What Can It Do?

- **Multi-Object Detection:** Identify multiple objects in a single image.
- **25 Real-World Classes:** Classify objects into everyday categories (Vehicles, Animals, etc.).
- **Visual Feedback:** Displays bounding boxes with confidence scores.
- **Model Comparison:** Compare predictions from multiple CNN architectures.
- **Interactive Dashboard:** Fully functional web app built with Streamlit.
- **Cloud Ready:** Optimized for easy deployment on Hugging Face Spaces.

---

## 🧠 Models Used

### Object Detection
* **YOLOv8** (Ultralytics) - *State-of-the-art real-time detection.*

### Image Classification
* **VGG16** - *Classic deep CNN architecture.*
* **ResNet50** - *Residual networks for deep feature extraction.*
* **MobileNetV2** - *Optimized for speed and mobile devices.*
* **EfficientNetB0** - *Balanced depth, width, and resolution for high accuracy.*

---

## 📂 Dataset

* **Source:** COCO 2017 (Custom Subset)
* **Size:** 2,500 Images (100 per class)
* **Structure:** Balanced classes to ensure fair training.
* **Context:** Real-world scenes containing multiple objects per frame.

---

## 💻 Application Pages

1.  **Home:** Project overview and usage guide.
2.  **Classification:** Single-object analysis using CNN models.
3.  **Detection:** Multi-object detection using YOLOv8.
4.  **Performance:** Metrics dashboard (Accuracy, F1-score, mAP, Inference Speed).
5.  **Webcam:** Live snapshot detection using your camera.

---

## 🛠️ Folder Structure

```bash
SmartVision_AI/
├── app.py                   # Main Streamlit Application
├── saved_models/            # Trained CNN model weights
├── yolo_runs/               # YOLO training outputs
├── smartvision_dataset/     # Dataset (Images & Labels)
├── scripts/                 # Training and utility scripts
├── inference_outputs/       # Saved results from detection
├── requirements.txt         # Python dependencies
└── README.md                # Documentation
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```
git clone https://github.com/<your-username>/SmartVision_AI.git
cd SmartVision_AI
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Install YOLOv8 (Ultralytics)

```
pip install ultralytics
```

---

## ▶️ Run Streamlit App

```
streamlit run app.py
```

App will open at:

```
http://localhost:8501
```

---

## 🏋️ Training Workflow

### 1️⃣ Classification Models
Each model has:
- Stage 1 → Train head with frozen backbone  
- Stage 2 → Unfreeze top layers + fine‑tune  

Scripts:
```
scripts/train_mobilenetv2.py
scripts/train_efficientnetb0.py
scripts/train_resnet50.py
scripts/train_vgg16.py
```

### 2️⃣ YOLO Training

```
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=50 imgsz=640
```

Outputs saved to:
```
yolo_runs/smartvision_yolov8s/
```

---

## 🧪 Supported Classes (25 COCO Classes)

```
airplane, bed, bench, bicycle, bird, bottle, bowl,
bus, cake, car, cat, chair, couch, cow, cup, dog,
elephant, horse, motorcycle, person, pizza, potted plant,
stop sign, traffic light, truck
```

---

## 🧰 Deployment on Hugging Face Spaces

You can deploy using **Streamlit SDK**.

### Steps:
1. Create public repository on GitHub  
2. Push project files  
3. Create new Hugging Face Space → select **Streamlit**  
4. Connect GitHub repo  
5. Add `requirements.txt`  
6. Enable **GPU** for YOLO (optional)  
7. Deploy 🚀  

---

## 🧾 requirements.txt Example

```
streamlit
tensorflow==2.13.0
ultralytics
numpy
pandas
Pillow
matplotlib
scikit-learn
opencv-python-headless
```

---

## 📄 .gitignore Example

```
saved_models/
*.h5
*.pt
*.weights.h5
yolo_runs/
smartvision_metrics/
inference_outputs/
__pycache__/
*.pyc
.DS_Store
env/
```

---

## 🙋 Developer

**SmartVision AI Project**  
Vignesh A
AI / Data Science Engineer
🔗 LinkedIn: https://www.linkedin.com/in/vignesh246v-ai-engineer/  

---

## 🏁 Conclusion

SmartVision AI integrates:
- Multi‑model classification  
- YOLO detection  
- Streamlit visualization  
- Full evaluation suite  

Perfect for:
- Research  
- Demonstrations  
- CV/AI portfolio  
- Real‑world image understanding  

---

Enjoy using SmartVision AI! 🚀🧠
=======
# Smart-vision-AI
>>>>>>> ef0986f82db7a8c8b3673315cf480bfe8b2aa437
