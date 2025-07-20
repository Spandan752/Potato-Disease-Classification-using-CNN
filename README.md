# 🌿 Plant Disease Detection - Potato Leaf Classifier

A deep learning-based system to detect diseases in potato plants using image classification. This project uses a Convolutional Neural Network (CNN) to identify whether a potato plant leaf is **Healthy**, affected by **Late Blight**, or **Early Blight**.

---

## 🧠 Project Overview

The goal of this project is to help farmers and agricultural experts identify diseases in potato plants through image classification. It uses a trained CNN model to classify uploaded leaf images into one of three categories:

- 🟢 Healthy
- 🟤 Early Blight
- ⚫ Late Blight

The model returns a confidence score for each prediction.

---

## 🖼️ How It Works

1. Upload a leaf image (JPEG/PNG).
2. The model processes the image using a CNN built with TensorFlow.
3. A classification result and prediction confidence score are returned.
4. Can be accessed via both **Streamlit UI** and **FastAPI endpoint**.

---

## ⚙️ Tech Stack

- **TensorFlow** – for building and training the CNN model
- **FastAPI** – for serving the model via a RESTful API
- **Streamlit** – for interactive web-based UI
- **Uvicorn** – ASGI server to run FastAPI
- **Pillow** – for image processing
- **python-multipart** – to handle file uploads in FastAPI
- **Matplotlib, NumPy** – for data visualization and processing

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection
```
### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API
```bash
uvicorn api:app --reload
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```

 Sample Output
✅ Predicted Class: Late Blight
📊 Confidence Score: 92.4%

