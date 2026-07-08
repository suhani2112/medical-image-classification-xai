# 🩺 AI Chest X-ray Diagnosis System with Explainable AI (Grad-CAM)
# ## 🌐 Live Demo

[Click here to try the app] https://medical-image-classification-xai-d8sgfdneksjgkf3dbx9mqi.streamlit.app/

An end-to-end Deep Learning application for automated pneumonia detection from Chest X-ray images using a Custom Convolutional Neural Network (CNN). The application provides explainable predictions using Grad-CAM visualizations and offers an interactive Streamlit interface with direct TensorFlow model inference. FastAPI backend code is also included for API-based deployment.

---

## 📌 Features

- 🧠 Custom CNN for Pneumonia Classification
- 🔥 Explainable AI using Grad-CAM
- ⚡ FastAPI Backend
- 🎨 Interactive Streamlit Frontend
- 📤 Upload Chest X-ray Images
- 📊 Confidence Score
- ⏱️ Processing Time
- 📄 PDF Report Generation
- 🖼️ Grad-CAM Overlay Download

---

## 🏗️ Project Architecture

```
                Chest X-ray Image
                        │
                        ▼
              Image Preprocessing
                        │
                        ▼
                  CNN Model
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
   Prediction                  Grad-CAM
         │                             │
         └──────────────┬──────────────┘
                        ▼
                 FastAPI Backend
                        │
                        ▼
              Streamlit Frontend
                        │
                        ▼
        Prediction + Confidence + PDF Report
```

---

## 🖼️ Application Preview

### Home Screen

![Home](screenshots/home.png)

---

### Prediction Result

![Prediction](screenshots/prediction.png)

---

### Grad-CAM Visualization

![GradCAM](screenshots/gradcam.png)

---

### Generated PDF Report

![PDF](screenshots/pdf_report.png)

---

## 📂 Project Structure

```
medical-image-classification-xai/

│
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── gradcam.py
│   ├── model_loader.py
│   ├── predict.py
│   └── preprocess.py
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_gradcam_explainability.ipynb
│
├── models/
│
├── reports/
│
├── results/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Technologies Used

### Machine Learning

- TensorFlow
- Keras
- NumPy
- OpenCV
- Grad-CAM

### Backend

- FastAPI
- Uvicorn

### Frontend

- Streamlit

### Others

- Pillow
- ReportLab

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/suhani2112/medical-image-classification-xai.git
```

Go to project directory

```bash
cd medical-image-classification-xai
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Backend

```bash
cd backend

python -m uvicorn app:app --reload
```

---

## ▶️ Run Frontend

```bash
streamlit run streamlit_app.py
```

---

## 📊 Model Output

The application predicts:

- 🟢 Normal
- 🔴 Pneumonia

For every prediction it also generates:

- Confidence Score
- Processing Time
- Grad-CAM Explainability
- PDF Medical Report

---

## 🔥 Explainable AI

Grad-CAM highlights the important regions of the Chest X-ray that contributed to the CNN's prediction, making the model more interpretable.

---

## 📈 Future Improvements

- Multi-class Chest Disease Classification
- Docker Support
- Cloud Deployment
- User Authentication
- PACS Integration
- DICOM Image Support
- Doctor Dashboard

---

## 👩‍💻 Author

**Suhani Gupta**

GitHub:
https://github.com/suhani2112

---

## 📄 License

This project is developed for educational and research purposes.