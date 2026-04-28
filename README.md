# 🫁 Pneumonia Detection System

> **A deep learning CNN that detects pneumonia from chest X-ray images with 90%+ accuracy — deployed as an interactive web application.**

---

## 🌐 Live Demo

> *Add your Streamlit app link here: `https://your-app.streamlit.app`*

![App Screenshot](assets/screenshot.png)

---

## 🧠 What It Does

Pneumonia kills over 2 million people annually, and early detection is critical. This system allows any user — including non-medical professionals — to upload a chest X-ray and instantly receive an AI-powered classification:

- ✅ **NORMAL** — No signs of pneumonia detected
- 🔴 **PNEUMONIA** — Signs of pneumonia detected with confidence score

---

## 🔍 How It Works

The system uses a **two-stage pipeline**:

```
User uploads image
        │
        ▼
┌──────────────────────┐
│   Filter Model       │
│   Is this a valid    │
│   chest X-ray?       │
└────────┬─────────────┘
    Yes  │       No → ❌ Rejected with message
         ▼
┌──────────────────────┐
│   CNN Classifier     │
│   PyTorch model      │
│   Normal vs          │
│   Pneumonia          │
└────────┬─────────────┘
         ▼
   Result + Confidence Score
```

The **filter model** is a key design decision — it prevents the classifier from making false predictions on irrelevant images (e.g., photos of food, selfies, etc.).

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | **90%+** |
| Model Architecture | CNN (PyTorch) |
| Dataset | Chest X-Ray Images (Kaggle) |
| Classes | Normal, Pneumonia |

---

## 🏗️ Project Structure

```
pneumonia-detection-cnn/
├── model/
│   ├── train.py          # Model training script
│   ├── model.py          # CNN architecture
│   └── filter_model.py   # X-ray validity filter
├── app.py                # Streamlit web app
├── requirements.txt
└── README.md
```

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Model Architecture | Convolutional Neural Network (CNN) |
| Web App | Streamlit |
| Data Processing | NumPy, Pillow |
| Visualisation | Matplotlib |
| Language | Python 3.11+ |

---

## 🛠️ Setup

```bash
git clone https://github.com/mustaphy666/pneumonia-detection-cnn.git
cd pneumonia-detection-cnn
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Dataset

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

---

## ⚠️ Disclaimer

This tool is for **educational purposes only** and is not a substitute for professional medical diagnosis.

---

## 👤 Author

**Saheed Mustapha Olatunji**
- GitHub: [@mustaphy666](https://github.com/mustaphy666)
- LinkedIn: [mustapha-saheed](https://www.linkedin.com/in/mustapha-saheed)
