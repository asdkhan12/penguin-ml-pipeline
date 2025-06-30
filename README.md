# penguin-ml-pipeline
A lightweight end-to-end machine learning demo that predicts the species of Palmer Archipelago penguins based on simple biometric measurements. It showcases the full supervised-learning workflow—data loading & cleaning, feature encoding, train/test splitting, model training (Logistic Regression & Decision Tree), and performance evaluation—plus an interactive Streamlit app for real-time prediction.

Table of Contents
Project Overview

Prerequisites

Installation

Usage

Data Preprocessing

Model Training & Evaluation

Streamlit App

Deployment

Results

Future Work

License

Project Overview
This repository demonstrates how to build and deploy a simple species classifier using the Palmer Archipelago penguin dataset from Kaggle. The notebook and scripts cover:

Data Loading & Cleaning

Reading CSV data

Dropping rows with missing values

Encoding categorical features (sex, island)

Feature Engineering

Mapping sex to binary

One-hot encoding island categories

Modeling

Splitting data into train/test sets

Training Logistic Regression and Decision Tree classifiers

Evaluating with accuracy, precision/recall/F1, and confusion matrices

Interactive Demo

A Streamlit app (app.py) for live prediction based on user inputs

Prerequisites
Python 3.7+

pip package manager

A local or Colab environment capable of running Jupyter notebooks

(Optional) A free Kaggle account if you wish to re-download the data

Installation
Clone the repository

bash
Copy code
git clone https://github.com/your-username/penguin-species-classifier.git
cd penguin-species-classifier
Create a virtual environment (recommended)

bash
Copy code
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate.bat       # Windows
Install dependencies

bash
Copy code
pip install -r requirements.txt
Download the dataset

Place penguins_size.csv and/or penguins_lter.xlsx into the project root

Or use the Kaggle API:

bash
Copy code
kaggle datasets download -d uciml/palmer-archipelago-antarctica-penguin-data 
unzip palmer-archipelago-antarctica-penguin-data.zip
Usage
Data Preprocessing
Open the Jupyter notebook or run the script to clean and prepare the data:

python
Copy code
# In a notebook cell or script
import pandas as pd

df = pd.read_csv("penguins_size.csv")
df_clean = (
    df
    .dropna()                                      # remove missing rows
    .assign(
        sex=lambda d: d["sex"].str.upper().map({"MALE":1,"FEMALE":0})
    )
    .pipe(pd.get_dummies, columns=["island"], drop_first=True)
)
Model Training & Evaluation
Within the same notebook or via train.py:

bash
Copy code
python train.py
This will:

Split data (80% train, 20% test)

Train Logistic Regression & Decision Tree

Print accuracy and classification reports

Display confusion matrices via Matplotlib/Seaborn

Streamlit App
Run the interactive app:

bash
Copy code
streamlit run app.py
Enter your own penguin measurements and get an instant species prediction.

Deployment
Streamlit Community Cloud

Push your code to GitHub (include app.py, best_model.pkl, feature_cols.pkl, requirements.txt).

On streamlit.io/cloud, deploy by linking your repo.

Flask API (optional)
See api.py for a simple JSON endpoint. Deploy on Heroku, AWS, or any Docker host.

Binder / Colab

Binder: Add environment.yml or requirements.txt and launch via https://mybinder.org

Colab: Add a badge to your README linking to the notebook in your GitHub.

Results
Accuracy
Both models typically achieve >95 % accuracy on the hold-out set.

Classification Report
Precision, recall, and F1-scores for Adelie, Chinstrap, and Gentoo exceed 0.95.

Confusion Matrices
Visual inspection shows almost all predictions on the diagonal, indicating very few misclassifications.

Future Work
Hyperparameter Tuning: Grid search or randomized search to optimize tree depth, regularization, etc.

Advanced Models: Compare Random Forests, SVMs, or Gradient Boosting.

Cross-Validation: Use k-fold CV to get more robust performance estimates.

Web Deployment: Build a Flask-based REST API or wrap in a Docker container for scalable deployment.

Dashboard: Create a Dash or Plotly app to visualize feature importances and model comparisons.

License
This project is released under the MIT License. Feel free to use, modify, and redistribute!











