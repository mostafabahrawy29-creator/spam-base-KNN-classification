# Spambase Email Classification using K-Nearest Neighbors (KNN)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning project evaluating the impact of **Feature Selection** on distance-based classification using the **UCI Spambase Dataset**. The project benchmarks a baseline KNN classifier (57 features) against an optimized KNN classifier (top 20 features) to mitigate the **Curse of Dimensionality**.

---

## Project Overview
- **Dataset:** [UCI Spambase Dataset](https://archive.ics.uci.edu/dataset/94/spambase) (4,601 instances, 57 continuous predictors, binary target `is_spam`).
- **Core Model:** K-Nearest Neighbors (`KNeighborsClassifier`).
- **Key Techniques:**
  - Exploratory Data Analysis (EDA) & standardized mean difference analysis.
  - Stratified 80/20 Train-Test splitting.
  - Feature standardization with `StandardScaler` (critical for distance-based learners).
  - 5-Fold Cross-Validation for optimal $k$ and inverse distance weighting.
  - Feature Selection via **Random Forest Gini Importance** and **Mutual Information**.
  - Systematic validation across dimensionalities ($D \in [3, 57]$).

---

## Results Summary

| Metric | Baseline KNN (All 57 Features) | Optimized KNN (Selected 20 Features) | Difference ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Feature Count ($D$)** | **57** | **20** | **-65.0% (Dimensionality reduced)** |
| **Optimal Neighbors ($k$)** | $k = 15$ | $k = 9$ | $-6$ |
| **Accuracy** | **91.75%** | **92.51%** | **+0.76%** |
| **Precision** | **90.65%** | **91.29%** | **+0.64%** |
| **Recall** | **88.15%** | **89.53%** | **+1.38%** |
| **F1-Score** | **89.39%** | **90.40%** | **+1.01%** |
| **False Positives (Ham $\to$ Spam)** | **33** | **31** | **-2 (Fewer lost good emails)** |
| **False Negatives (Spam $\to$ Ham)** | **43** | **38** | **-5 (Fewer spam leaks)** |
| **Inference Latency** | **31.08 ms** | **20.60 ms** | **-33.7% (Faster throughput)** |

---

## Repository Structure

```text
├── Spambase_KNN_Classification.ipynb  # Interactive step-by-step Jupyter Notebook
├── run_full_analysis.py               # Complete standalone automated pipeline
├── build_notebook.py                  # Script to programmatically generate notebook
├── spambase.data                      # UCI Spambase raw data file
├── spambase.names                     # UCI attribute definitions and names
├── spambase.DOCUMENTATION             # Documentation and background notes
├── requirements.txt                   # Required Python libraries
├── .gitignore                         # Standard git ignore file
└── outputs/                           # Output plots and summary CSV
    ├── eda_class_distribution.png
    ├── eda_feature_differences.png
    ├── eda_key_features_boxplots.png
    ├── eda_correlation_heatmap.png
    ├── feature_importances_comparison.png
    ├── feature_selection_curve.png
    ├── confusion_matrix_comparison.png
    ├── metrics_comparison_barchart.png
    ├── roc_curve_comparison.png
    └── model_comparison_summary.csv
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/USERNAME/Spambase-KNN-Classification.git
cd Spambase-KNN-Classification
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run the Automated Analysis
```bash
python run_full_analysis.py
```

### 4. Run the Jupyter Notebook
```bash
jupyter notebook Spambase_KNN_Classification.ipynb
```

---

## Key Engineering Takeaways
1. **Curse of Dimensionality in KNN:** In 57-dimensional space, Euclidean distance metrics degrade as data becomes sparse and equidistant. Reducing dimensions from 57 to 20 improved accuracy, recall, and precision simultaneously.
2. **Standardization is Crucial:** Distance-based models are extremely sensitive to magnitude disparities (e.g., word frequencies in `[0, 100]` vs. capital run lengths in `[1, 10000]`). Standardization guarantees uniform feature attribution.
3. **Efficiency Gains:** Removing 65% of features reduced test-set inference latency by **33.7%**, critical for high-throughput email filtering environments.

---

## License
This project is open-sourced under the [MIT License](LICENSE).
