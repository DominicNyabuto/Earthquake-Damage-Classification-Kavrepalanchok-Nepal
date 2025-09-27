# Earthquake Damage Classification - Kavrepalanchok, Nepal 

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/sklearn-latest-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-latest-yellow.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Predicting earthquake damage severity in Nepalese buildings using machine learning classification algorithms

## Project Overview

This project analyzes over 76,000 building records to predict earthquake damage severity (Grades 1-5) using building characteristics and multiple classification algorithms (logistic regression, decision trees, LightGBM). The best model (LightGBM) achieved **98.89% recall** for severe damage detection, making it highly effective for disaster response planning.

## Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **LightGBM** | **0.6388** | **0.6044** | **0.9889** | **0.7503** |
| Decision Tree | 0.6304 | 0.5983 | 0.9933 | 0.7468 |
| Logistic Regression | 0.6378 | 0.6197 | 0.8799 | 0.7272 |

*LightGBM emerged as the best model, achieving exceptional recall for identifying severely damaged buildings.*

## Quick Start

1. Clone the repo
   ```bash
   git clone https://github.com/DominicNyabuto/Earthquake-Damage-Classification-Kavrepalanchok-Nepal.git
   cd Earthquake-Damage-Classification-Kavrepalanchok-Nepal

2. Install dependencies
   ```bash
   pip install -r requirements.txt
3. Run the notebook
   ```bash
   jupyter notebook notebooks/earthquake-damage-in-nepal-classification.ipynb

## Dataset (kavrepalanchok_raw.csv) Features

The dataset includes **16 building characteristics**:
- **Structural**: Foundation type, roof type, superstructure material
- **Physical**: Age, height, floor count, area
- **Environmental**: Land surface condition, position
- **Post-earthquake**: Condition assessment

**Target Variable**: binary target severe_damage (0-1) was derived from damage_grade (Grade 1–5), where 0 = less severe (Grades 1–3) and 1 = severe (Grade 4-5).

## Methodology

### Data Preprocessing
- **Label Encoding** for categorical variables (efficient for moderate cardinality)
- **StandardScaler** for numerical features
- **Binary Classification** (severe_damage column with 0 and 1 as values for severe and less severe damage)

### Model Selection
1. **Baseline Logistic Regression**
2. **Tuned Logistic Regression** with GridSearchCV
3. **Decision Tree** with hyperparameter optimization
4. **LightGBM** (final model) - chosen for speed and categorical handling

### Evaluation Strategy
- **Recall-focused metrics** (critical for disaster response)
- **5-fold Cross-validation**
- **Comprehensive classification reports**

## Model Performance Visualization

![Model Comparison](results/figures/model_comparison.png)
![Feature Importance](results/figures/feature_importance_lightgbm.png)
![Confusion Matrix](results/figures/confusion_matrix_lightgbm.png)

## Repository Structure
├── notebooks/ # Jupyter notebooks for analysis
├── data/ # Dataset files
├── models/ # Trained model files
├── results/ # Visualizations and reports
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore

earthquake-damage-classification-nepal/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01-data-exploration.ipynb
│   ├── 02-data-preprocessing.ipynb
│   ├── 03-model-training.ipynb
│   └── earthquake-damage-in-nepal-classification.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── models/
│   ├── saved_models/
│   └── model_comparison.csv
├── results/
│   ├── figures/
│   ├── classification_reports/
│   └── confusion_matrices/
├── docs/
│   ├── methodology.md
│   └── model_performance.md
└── .github/
    ├── ISSUE_TEMPLATE/
    └── workflows/

