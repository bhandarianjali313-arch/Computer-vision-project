# AI-Based Real-Time Industrial Defect Detection

An AI-powered Computer Vision system designed to detect and localize defects on industrial surfaces using Deep Learning and YOLO. The system provides an interactive web dashboard where users can upload inspection images, detect potential defects, view confidence scores, and visualize detected regions using bounding boxes.

## 🚀 Project Overview

Traditional industrial quality inspection often requires manual examination, which can be time-consuming and prone to human error. This project aims to automate surface-defect inspection using computer vision and deep learning.

The system processes industrial images through a trained YOLO model and returns the detected defect classes, confidence scores, bounding-box coordinates, and annotated inspection images.

## 🎯 Objectives

- Automate industrial surface-defect detection.
- Reduce dependence on manual visual inspection.
- Identify and localize potential defects in images.
- Display defect categories and confidence scores.
- Provide an interactive dashboard for inspection results.
- Support faster and more consistent quality-control workflows.

## ✨ Key Features

- **AI-Based Defect Detection:** Uses a YOLO object-detection model to identify defects.
- **Image Upload:** Supports JPG, JPEG, PNG, and WEBP images.
- **Defect Classification:** Displays the predicted defect class.
- **Confidence Score:** Shows the model's confidence for each detection.
- **Bounding-Box Localization:** Highlights detected regions in the image.
- **Annotated Images:** Displays the processed image with detection boxes.
- **Inference-Time Measurement:** Reports the time taken for model inference.
- **REST API:** Connects the frontend to the computer-vision backend.
- **Interactive Dashboard:** Provides a simple interface for uploading images and reviewing results.
- **API Documentation:** Supports interactive testing through FastAPI Swagger UI.

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, Vite, Tailwind CSS |
| Backend | Python, FastAPI |
| Computer Vision | OpenCV, NumPy |
| Deep Learning | YOLO, Ultralytics |
| API Communication | REST API, Axios |
| Model Format | PyTorch `.pt` |
| Development Tools | VS Code, Git, GitHub |

## 🏗️ System Architecture

```text
Industrial Surface Image
          |
          v
    React Dashboard
          |
          v
     FastAPI Backend
          |
          v
    Image Preprocessing
          |
          v
      YOLO Model
          |
          v
    Defect Detection
          |
     +----+----+
     |         |
     v         v
 Defect Class  Confidence Score
     |         |
     +----+----+
          |
          v
   Bounding-Box Drawing
          |
          v
 Annotated Image + JSON Results
          |
          v
     React Dashboard
```

## 📁 Project Structure

```text
AI-Industrial-Defect-Detection/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── detector.py
│   │   ├── config.py
│   │   └── utils.py
│   │
│   ├── models/
│   │   └── best.pt
│   │
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── Navbar.jsx
    │   │   ├── UploadPanel.jsx
    │   │   ├── DetectionResult.jsx
    │   │   ├── StatsCard.jsx
    │   │   └── Loading.jsx
    │   │
    │   ├── App.jsx
    │   ├── main.jsx
    │   └── index.css
    │
    ├── public/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── postcss.config.js
```

*Note: Adjust the folder structure above to match your actual GitHub repository. The `best.pt` file must be supplied by training or otherwise obtaining a compatible trained YOLO model.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Industrial-Defect-Detection
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's actual URL.

### 2. Set Up the Backend

Navigate to the backend folder:

```bash
cd backend
```

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Add the YOLO Model

Place your compatible trained YOLO model here:

```text
backend/models/best.pt
```

The model's class names and detection format must match the backend's expectations.

### 4. Start the Backend

Run the following command from the backend directory:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Set Up the Frontend

Open a second terminal and navigate to the frontend folder:

```bash
cd frontend
```

Install the dependencies:

```bash
npm install
```

Create a `.env` file in the frontend root:

```env
VITE_API_URL=http://127.0.0.1:8000
```

Start the development server:

```bash
npm run dev
```

Open the local frontend URL displayed in the terminal, usually:

```text
http://localhost:5173
```

Keep both the frontend and backend running during local testing.

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Returns project and service information |
| GET | `/health` | Checks backend and model status |
| GET | `/classes` | Returns the loaded model's class names |
| POST | `/predict` | Detects defects and returns JSON results |
| POST | `/predict/image` | Returns an annotated JPEG image |
| GET | `/metrics` | Returns basic service status information |

### Example Prediction Response

```json
{
  "request_id": "example-request-id",
  "filename": "industrial_surface.jpg",
  "image": {
    "width": 640,
    "height": 480,
    "channels": 3
  },
  "detection_count": 1,
  "inference_time_ms": 20.5,
  "detections": [
    {
      "class_id": 0,
      "class_name": "example_defect",
      "confidence": 0.94,
      "bbox": {
        "x1": 120,
        "y1": 80,
        "x2": 260,
        "y2": 210
      }
    }
  ]
}
```

*The response above is illustrative, not a measured result. Actual classes, confidence scores, coordinates, and inference times depend on your trained model and hardware.*

## 🔄 Working Process

1. The user opens the web dashboard.
2. The user uploads an industrial surface image.
3. The frontend sends the image to the FastAPI backend.
4. OpenCV decodes the image for processing.
5. The YOLO model performs inference.
6. The backend extracts defect classes, confidence scores, and bounding boxes.
7. The system generates an annotated image.
8. The frontend displays the inspection results.

## 🧪 Testing

1. Start the backend and frontend.
2. Open the dashboard in your browser.
3. Upload a supported industrial image.
4. Click **Detect Defects**.
5. Review the detection count, defect classes, confidence scores, bounding boxes, and annotated image.

For direct API testing, use the Swagger documentation at `http://127.0.0.1:8000/docs`.

## 📌 Applications

- Steel and metal surface inspection
- Manufacturing quality control
- Automated production-line inspection
- Industrial material defect identification
- Surface-quality monitoring
- Computer-vision research and education

## 🔮 Future Enhancements

- Real-time camera and video-stream inspection
- Integration with industrial cameras
- Defect severity classification
- Inspection history and database storage
- Automated quality-control reports
- Real-time alerts for detected defects
- Production-line and PLC integration
- Model optimization for edge devices

## ⚠️ Limitations

- Detection performance depends on the quality and coverage of the training dataset.
- A compatible trained YOLO model is required for actual inference.
- Incorrect lighting, image blur, and unfamiliar defect types can affect predictions.
- Confidence scores are model estimates, not guarantees of correctness.
- Real-time production performance depends on the hardware and deployment environment.

## 👥 Contributors

Add the names of your project team members here.

## 📄 License

This project is intended for educational and research purposes. Add an appropriate open-source license before distributing the repository.

---

**AI-Based Real-Time Industrial Defect Detection**  
*Automating quality inspection through Computer Vision and Deep Learning.*
