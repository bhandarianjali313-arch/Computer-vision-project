# AI-Based Real-Time Industrial Defect Detection

A computer-vision system for detecting and classifying metal surface defects using YOLO, a FastAPI backend, and a React frontend.

## Features

- Detects and classifies six categories of metal surface defects.
- Provides prediction APIs through FastAPI.
- Supports image predictions with annotated output.
- Provides health-check and defect-class information endpoints.
- Includes an ML pipeline, configuration files, and automated tests.
- Includes a React/Vite frontend for interacting with the system.
- Provides application metrics through Prometheus-compatible instrumentation.

## Defect Classes

| Class ID | Defect Name |
|---:|---|
| 0 | Crazing |
| 1 | Inclusion |
| 2 | Patches |
| 3 | Pitted Surface |
| 4 | Rolled-in Scale |
| 5 | Scratches |

## Technology Stack

| Component | Technologies |
|---|---|
| Machine Learning | Python, PyTorch, Ultralytics YOLO |
| Computer Vision | OpenCV, NumPy |
| Backend API | FastAPI, Uvicorn |
| Frontend | React, Vite |
| Styling | Tailwind CSS |
| Monitoring | Prometheus client |
| Testing | Pytest |
| Deployment Configuration | Docker, Docker Compose |

## Repository Structure

```text
Computer-vision-project/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── detector.py
│   ├── utils.py
│   └── services/
├── ml/
│   ├── configs/
│   ├── scripts/
│   ├── src/
│   └── tests/
├── src/                   # React frontend
├── models/                # Local model weights; excluded from Git
├── data/                  # Local datasets; excluded from Git
├── requirements.txt
├── requirements-ml.txt
├── package.json
├── package-lock.json
├── Dockerfile
├── docker-compose.yml
├── vite.config.js
├── .env.example
└── README.md
```

## Prerequisites

Install the following software before running the project:

- Python
- Node.js and npm
- Git
- Trained model weights required for inference

## Backend Setup

Run the following commands from the repository root in PowerShell.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If the virtual environment already exists, activate it rather than creating it again.

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Verify model availability

The application is configured to use the following model path by default:

```text
models/optimized/yolo_neu_optimized_best.pt
```

Ensure that the required trained model file exists at the configured path before running predictions.

Model weights and datasets are excluded from Git and must be obtained separately.

### 4. Start the backend

```powershell
uvicorn app.main:app --reload
```

The API should be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Frontend Setup

Open a second PowerShell terminal in the repository root.

### 1. Install frontend dependencies

```powershell
npm ci
```

### 2. Configure the API URL

The example environment configuration contains:

```text
VITE_API_URL=http://127.0.0.1:8000
```

Create a local `.env` file from `.env.example` if one does not already exist:

```powershell
Copy-Item .env.example .env
```

If `.env` already exists, keep it and verify that its API URL is correct.

### 3. Start the frontend

```powershell
npm run dev
```

Open the local URL printed by Vite in your terminal.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Returns service information |
| GET | `/health` | Checks application health |
| GET | `/classes` | Returns the available defect classes |
| POST | `/predict` | Returns prediction results |
| POST | `/predict/image` | Returns an annotated prediction image |
| GET | `/metrics` | Exposes application metrics |

For request parameters, payload formats, and response schemas, refer to the interactive API documentation at `/docs`.

## Running Tests

Activate the project's Python virtual environment and run:

```powershell
python -m pytest ml/tests -q
```

The current test run completed with **110 passed tests**. Two FastAPI deprecation warnings were reported for the older `on_event` lifecycle API.

## Building the Frontend

To create a production frontend build, run:

```powershell
npm run build
```

The generated production assets are written to the `dist/` directory.

The current frontend build completed successfully.

## Model Evaluation and Performance

Model performance depends on the trained weights, evaluation dataset, hardware, and inference configuration.

Record and report evaluation metrics using the actual results produced by the project's evaluation pipeline. Do not interpret benchmark values as guaranteed production performance.

## Deployment Notes

- Keep trained model weights and datasets out of Git when they are excluded by `.gitignore`.
- Provide the required model weights to the deployment environment.
- Configure the frontend API URL for the target backend environment.
- Do not commit secrets, credentials, or private environment configuration.
- Validate Docker builds in an environment where Docker is installed.
- Confirm that the model path mounted into a container matches the path expected by the application.

## Current Verification

The following checks have completed successfully:

- **ML tests:** 110 passed.
- **Frontend production build:** successful.

Docker image building has not been verified in the current environment.

## License

Add the applicable project license before distributing or publishing the repository.