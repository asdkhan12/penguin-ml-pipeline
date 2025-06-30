# Penguin Species Classifier

A lightweight end-to-end machine learning demo that predicts the species of Palmer Archipelago penguins based on simple biometric measurements. It covers the full supervised-learning workflow—data loading & cleaning, feature encoding, train/test splitting, model training (Logistic Regression & Decision Tree), and performance evaluation—plus an interactive Streamlit app for real-time prediction.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Data Preprocessing](#data-preprocessing)  
  - [Model Training & Evaluation](#model-training--evaluation)  
  - [Streamlit App](#streamlit-app)  
- [Deployment](#deployment)  
- [Results](#results)  
- [Future Work](#future-work)  
- [License](#license)  

---

## Project Overview

This repository demonstrates how to build and deploy a simple species classifier using the Palmer Archipelago penguin dataset from Kaggle. The notebook and scripts cover:

1. **Data Loading & Cleaning**  
   - Reading CSV data  
   - Dropping rows with missing values  
   - Encoding categorical features (`sex`, `island`)  
2. **Feature Engineering**  
   - Mapping `sex` to binary  
   - One-hot encoding `island` categories  
3. **Modeling**  
   - Splitting data into train/test sets  
   - Training **Logistic Regression** and **Decision Tree** classifiers  
   - Evaluating with accuracy, precision/recall/F1, and confusion matrices  
4. **Interactive Demo**  
   - A Streamlit app (`app.py`) for live prediction based on user inputs  

---

## Prerequisites

- Python 3.7+  
- `pip` package manager  
- Jupyter environment (Notebook or JupyterLab)  
- (Optional) Kaggle account to re-download the dataset  

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/penguin-species-classifier.git
   cd penguin-species-classifier

python3 -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate.bat      # Windows

pip install -r requirements.txt

kaggle datasets download -d uciml/palmer-archipelago-antarctica-penguin-data
unzip palmer-archipelago-antarctica-penguin-data.zip

import pandas as pd

df = pd.read_csv("penguins_size.csv")
df_clean = (
    df
    .dropna()  # remove missing rows
    .assign(
        sex=lambda d: d["sex"].str.upper().map({"MALE":1,"FEMALE":0})
    )
    .pipe(pd.get_dummies, columns=["island"], drop_first=True)
)


python train.py

