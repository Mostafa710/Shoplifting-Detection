# 🛒 Shoplifting Detection

## 📌 Project Overview
This project implements a **deep learning system for shoplifting detection** using **3D Convolutional Neural Networks (3D CNNs)**. The system analyzes video footage from supermarket surveillance cameras to detect suspicious shoplifting activity in real-time.

---

## 🚀 Features
- Video-based analysis using **3D CNNs** to capture spatial and temporal patterns.
- Preprocessing pipeline for converting raw CCTV footage into model-ready clips.
- Trained model to classify **normal vs. shoplifting activity**.
- Deployment-ready using **Django** with a clean web interface.
- Virtual environment setup for reproducibility.

---

## 🗂️ Project Structure
```
├── shoplifting-detection-pretrained.ipynb    # Kaggle notebook (model training)
├──venv/                                      # Virtual environment
|   ├── Scripts/                              # Windows executables
|   │   └── activate                          # Activate script
|   ├── Lib/                                  # Installed site-packages
|   └── pyvenv.cfg
├──shoplifting_site/                          # Main Django project folder
|   │
|   ├── detection/                            # App folder (handles detection logic & frontend)
|   │   ├── model/                            # Pretrained DL model file
|   │   │   └── best_3D_CNN_model_original_data.pth
|   │   │
|   │   ├── static/                           # Static assets (CSS, JS, images)
|   │   │   └── css/
|   │   │       └── style.css
|   │   │
|   │   ├── templates/                        # HTML templates
|   │   │   ├── base.html
|   │   │   ├── index.html
|   │   │   ├── result.html
|   │   │   └── about.html
|   │   │
|   │   ├── inference.py                      # Model inference code (loads model + prediction function)
|   │   ├── urls.py                           # App-specific URL routes
|   │   ├── views.py                          # Handles requests (upload, inference, render templates)
|   │   └── __init__.py
|   │
|   ├── media/                                # Uploaded media files
|   │   └── uploads/                          # Uploaded videos for detection
|   │
|   ├── shopliftingsite/                      # Django project core files
|   │   ├── __init__.py
|   │   ├── asgi.py
|   │   ├── settings.py                       # Project settings (add 'detection' app, static/media config)
|   │   ├── urls.py                           # Root URLs (include detection.urls)
|   │   └── wsgi.py
|   │
|   ├── manage.py                             # Django management script
|   └── requirements.txt                      # Dependencies (torch, torchvision, Django, etc.)
└──
```

---

## ⚙️ Installation & Usage
### 1️⃣ Clone the repository
```
git clone https://github.com/Mostafa710/Shoplifting-Detection.git
cd shoplifting-detection
```

### 2️⃣ Create and activate a virtual environment
`virtualenv venv` or `python -m venv venv`
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

### 3️⃣ Install dependencies
`pip install -r requirements.txt`

### 4️⃣ Apply migrations
`python manage.py migrate`

### 5️⃣ Run the server
`python manage.py runserver`
Then open http://127.0.0.1:8000/ in your browser.

---

## 🧠 Model Training
- Model: **3D CNN** trained on supermarket surveillance dataset.
- Input: Short video clips from CCTV streams.
- Output: `0` → Normal activity, `1` → Shoplifting activity.

---

## 📸 Screenshots

### 🏠 Homepage
<img width="1114" height="595" alt="Screenshot 2025-08-22 150321" src="https://github.com/user-attachments/assets/e821902c-fbc9-4465-9142-8f636338d6b0" />


### ℹ️ About Page
<img width="1116" height="594" alt="Screenshot 2025-08-22 150541" src="https://github.com/user-attachments/assets/23bd685b-32ec-4410-b194-9d60fc873d3b" />

---

## 🛠️ Tech Stack
- **Backend:** Django (Python)  
- **Frontend:** HTML, CSS, Bootstrap  
- **Database:** SQLite
- **ML Model:** 3D CNN (Shoplifting Detection)  
- **Deployment Environment:** Virtualenv

---

## 📬 Contact
For questions or collaboration, feel free to connect:

[LinkedIn](https://www.linkedin.com/in/mostafa-mamdouh-80b110228) | [Email](mailto:mostafamamdouh710@gmail.com)

---




