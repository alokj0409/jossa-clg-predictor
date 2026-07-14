# JoSAA ML-Based College Predictor 

A high-performance machine learning pipeline and interactive dashboard that predicts JoSAA (Joint Seat Allocation Authority) seat allotment outcomes using 6 years of historical data (2020-2025).

## 🚀 Key Features
1. **Dual-Model Pipeline**: Uses separate XGBoost models for **IITs** (JEE Advanced ranks) and **NITs/IIITs/GFTIs** (JEE Main ranks) to handle scale differences.
2. **Admission Probability Classifier**: Sampler expands historical cutoffs into a binary dataset to train an **XGBoost Classifier**, outputting real-time probability estimates (e.g. "87.5% Chance of Admission").
3. **Cutoff Regressor**: Trains an **XGBoost Regressor** to predict expected round 6 closing ranks directly.
4. **Historical Mean target encoding**: Automatically matches high-cardinality institute/program pairs with their historical demand averages.
5. **Interactive UI**: Sleek, glassmorphic Streamlit dashboard with Home State quota filtering, dynamic category checks, and recommendation tabs (**Safe / Reach / Dream**).

---

## 🛠️ Project Structure
```text
josaa-ml-predictor/
├── data/
│   ├── raw/                 # Raw dataset (merged_allround.csv)
│   └── processed/           # Processed & sampled training data
├── src/
│   ├── download_data.py     # Database downloader & verification
│   ├── data_pipeline.py     # Cleansing, splitting & sampling
│   └── train_models.py      # XGBoost models trainer & exporter
├── api/
│   └── main.py              # FastAPI server backend
├── app/
│   └── streamlit_app.py     # Streamlit premium frontend
├── models/                  # Exported .pkl models & mappings
├── requirements.txt         # Project dependencies
└── run.py                   # Central orchestrator CLI
```

---

## 💻 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess & Sample Data
Runs data cleaning, historical feature creation, and candidate rank sampling:
```bash
python run.py pipeline
```

### 3. Train Models
Trains the regressors/classifiers and generates lookup maps for the API:
```bash
python run.py train
```

### 4. Start Servers
Starts the FastAPI API (port 8000) and launches the Streamlit app:
```bash
python run.py start
```
Navigate to `http://localhost:8501` to use the app.

---

## 📊 Technical Performance
- **Validation Strategy**: Time-based split. Trained on **2020-2024**, tested on **2025** to represent true forecasting capability.
- **Features Used**: `institute`, `program`, `quota`, `category`, `gender`, `hist_crank_mean`, `hist_crank_min`, `hist_crank_max`, `candidate_rank`, `rank_diff`, `rank_ratio`.
