# 🌊 AquaPredict — Industrial Wastewater Pollution Prediction

A college mini-project demonstrating **8 Unit 5 Machine Learning algorithms** applied to industrial wastewater quality prediction.

## ✨ Features

- **8 ML Algorithms**: CART Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting, ABM (MLP-based), GAM (Generalized Additive Model), Ensemble Learning (Voting + Stacking)
- **Interactive Dashboard**: Modern, dark-sidebar Streamlit UI with teal/blue accents
- **Dataset Flexibility**: Built-in synthetic demo data + CSV upload support
- **Real ML Training**: Actual scikit-learn models — no hardcoded results
- **Visualizations**: Plotly & Matplotlib charts for every algorithm
- **Model Comparison**: Side-by-side performance comparison of all models
- **Live Prediction**: Enter wastewater measurements and get instant predictions

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| ML | scikit-learn, pyGAM |
| Data | pandas, NumPy |
| Charts | Plotly, Matplotlib, Seaborn |

## 🚀 Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Open in browser
# → http://localhost:8501
```

## 📁 Project Structure

```
ML3/
├── app.py                     # Main Streamlit application (all 12 pages)
├── real_wastewater_clean.csv  # Real UCI Wastewater Treatment Plant dataset (10 core variables)
├── real_wastewater_data.csv   # Real UCI Wastewater Treatment Plant dataset (full 39 features)
├── sample_wastewater_data.csv # Synthetic wastewater demo dataset (600 samples)
├── requirements.txt           # Python dependencies
├── run.bat                    # 1-Click launcher for Windows
└── README.md                  # Project documentation & setup instructions
```

## 🌿 Real Dataset Information

- **Source**: UCI Machine Learning Repository — *Water Treatment Plant Dataset*
- **Official Dataset Link**: [https://archive.ics.uci.edu/dataset/107/water+treatment+plant](https://archive.ics.uci.edu/dataset/107/water+treatment+plant)
- **Records**: 527 actual daily operational sensor measurements from an urban wastewater treatment plant
- **Key Features**: Flow rate (`Flow_m3_d`), pH (`pH`), Biological Oxygen Demand (`BOD_mg_L`), Chemical Oxygen Demand (`COD_mg_L`), Total Suspended Solids (`TSS_mg_L`), Sediments (`Sediments_mL_L`), Conductivity (`Conductivity_uS_cm`), Zinc concentration (`Zinc_mg_L`), Effluent output (`Output_COD_mg_L`)
- **Target Options**:
  - Classification: `Pollution_Level` (`Low`, `Moderate`, `High` derived via multi-parameter Comprehensive Pollution Index)
  - Regression: `COD_mg_L`, `BOD_mg_L`, or `Output_COD_mg_L`

## 📊 Model Accuracy Benchmark (85%+ on All Models)

| # | Algorithm | Category | Accuracy | Status |
|---|---|---|:---:|:---:|
| 1 | **CART Decision Tree** | Decision Tree | **87.74%** | ✅ PASS (85%+) |
| 2 | **Bagging** | Bagging Ensemble | **85.85%** | ✅ PASS (85%+) |
| 3 | **Random Forest** | Random Forest | **86.79%** | ✅ PASS (85%+) |
| 4 | **AdaBoost** | Boosting | **88.68%** | ✅ PASS (85%+) |
| 5 | **Gradient Boosting** | Boosting | **91.51%** | ✅ PASS (85%+) |
| 6 | **ABM (MLP-based)** | Adaptive Basis Model | **99.06%** | ✅ PASS (85%+) |
| 7 | **GAM (Spline Approx.)** | Generalized Additive | **89.62%** | ✅ PASS (85%+) |
| 8 | **Voting Ensemble** | Meta-Ensemble | **93.40%** | ✅ PASS (85%+) |
| 9 | **Stacking Ensemble** | Meta-Ensemble | **89.62%** | ✅ PASS (85%+) |

## 📖 How to Use

1. **Overview** — See project summary and key metrics
2. **Dataset & Preprocessing** — Upload CSV or use the built-in synthetic demo data, select features & target, prepare train/test split
3. **Model Pages** (CART → Ensemble) — Train each algorithm, view metrics & visualizations
4. **Model Comparison** — Compare all trained models side-by-side
5. **Test Prediction** — Enter wastewater values and predict pollution level

## 📝 Notes

- The built-in demo data is **synthetic** and clearly labelled as such
- Pollution level thresholds in the demo data are **data-driven quantiles**, not official standards
- All models are trained in real-time using scikit-learn — no fake results
- For best results, upload a real wastewater quality dataset in CSV format

## 👥 Team

College Mini Project — Unit 5: Machine Learning Algorithms

---
*Built with ❤️ using Streamlit & scikit-learn*
