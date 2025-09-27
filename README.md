# 🏗️ Earthquake Damage Classification - Kavrepalanchok, Nepal 

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/sklearn-latest-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-latest-yellow.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Predicting earthquake damage severity in Nepalese buildings using machine learning classification algorithms

## 🎯 Project Overview

This project analyzes over 76,000 building records to predict earthquake damage severity (Grades 1-5) using building characteristics and multiple classification algorithms (logistic regression, decision trees, LightGBM). The best model (LightGBM) achieved **98.89% recall** for severe damage detection, making it highly effective for disaster response planning.

## 📊 Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **LightGBM** | **0.6388** | **0.6044** | **0.9889** | **0.7503** |
| Decision Tree | 0.6304 | 0.5983 | 0.9933 | 0.7468 |
| Logistic Regression | 0.6378 | 0.6197 | 0.8799 | 0.7272 |

*LightGBM emerged as the best model, achieving exceptional recall for identifying severely damaged buildings.*

## 🚀 Quick Start

