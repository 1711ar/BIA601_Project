# BIA601_Project – Intelligent Algorithms

## Overview
This project implements a Genetic Algorithm (GA) for feature selection.  
It reduces dataset dimensionality while maintaining model accuracy.

---

## Components
| File | Description |
|------|--------------|
| ga_feature_select.py | Implementation of the Genetic Algorithm for feature selection. |
| run_experiment.py | Runs the GA on the Iris dataset. |
| compare_baselines.py | Compares GA results with traditional methods (PCA, SelectKBest). |
| webapp_streamlit.py | Web interface for visualization (Streamlit). |
| data/ | Contains datasets used for experiments. |

---

## Technologies
- Python 3.10+
- scikit-learn
- numpy, pandas
- streamlit

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
