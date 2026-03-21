# PPG Heart Rate Prediction (1D Convolutional Neural Network)

**Author**: Rohit  
**Tech Stack**: Python, TensorFlow / Keras, Flask, HTML/CSS dashboard, NumPy, Pandas  

---

## Overview
This application serves as an end-to-end framework for predicting Heart Rate (BPM) from continuous Photoplethysmogram (PPG) signals. It utilizes a **1D Convolutional Neural Network (CNN)** optimized for time-series feature extraction and presents predictions on a premium visual dashboard using Flask API endpoints.

---

## 🔬 Model & Pipeline Optimization
The training pipeline (`main.py`) processes indices from the **BIDMC PPG dataset** with optimized robustness guidelines:

1.  **Overlapping Window Lookback**: Standardized 8-second time frames (1000 nodes at 125Hz) preceding every 1Hz HR reading frequency.
2.  **Anti-Leakage Validation**: The ETL steps partition train/test arrays *prior* to augmenting vectors with synthetic Gaussian noise and shifts to prevent over-optimistic evaluation metrics.
3.  **Huber Loss Metric**: Utilized to suppress outliers and prioritize core waveform trends.

---

## 📂 Project Structure
*   `main.py`: ETL scripts and Conv1D algorithm compilation, scoring, and saving procedures.
*   `app.py`: Backend Flask route handlers loading `hr_model.keras` into inference.
*   `templates/`: UI scripts rendered via vanilla glassmorphism structures connecting endpoints accurately with Chart.js controllers on trigger query loads.

---

## ⚙️ Usage Breakdown

### 1. Training Cycle
To recreate weights utilizing robust ETL scripts:
```bash
python main.py
```

### 2. Frontend Query loads
To evaluate inference routing procedures simply run:
```bash
python app.py
```
Open **`http://localhost:5000`** and drag in single column sequential datasets matching payload structures to test visual trigger metrics.

---

## 🧪 Running Unit Tests
Validate endpoint logic and shape validation safety checks using `pytest`:
```bash
python -m pytest test_app.py
```

---

## 🐳 Containerization (Docker)
To package and run the Dashboard inside an isolated environment:

1.  **Build the Docker Image**:
    ```bash
    docker build -t ppg-dashboard .
    ```

2.  **Run the Container**:
    ```bash
    docker run -p 5000:5000 ppg-dashboard
    ```
    *The Flask server binds with safety routes across Generic Host binds triggers handles.*
