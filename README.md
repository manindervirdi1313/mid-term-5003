---
title: Maple Physio Appointment Risk Predictor
emoji: hospital
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: 3.12
pinned: false
---

# Maple Physio Appointment Risk Predictor

## What It Does

Maple Physio Appointment Risk Predictor estimates the probability that a patient will miss a physiotherapy appointment.

The application helps clinic staff identify appointments that may need follow-up contact. It uses information available at booking time and recommends whether the patient should be contacted.

The model uses a threshold selected through out-of-fold predictions on the training data.

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/manindervirdi1313/mid-term-5003.git
cd mid-term-5003
```

### 2. Create and activate the Conda environment

```bash
conda env create -f environment.yml
conda activate midterm-5003
```

Alternatively, install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

### 3. Train the model

```bash
python -m noshows.train
```

This creates the trained model and model information files in the `models/` directory and evaluation reports in the `reports/` directory.

### 4. Launch the application

```bash
python app.py
```

Open the local Gradio address displayed in the terminal, usually:

`http://127.0.0.1:7860`

## Team

This implementation was completed by Maninder Singh Virdi.

The project was developed and tested locally using the assigned project dataset and requirements.

## Decisions

### Data and Features

- Used the supplied appointment dataset generator without modifying it.
- Used a stratified 75/25 training and test split.
- Used random seed 11 for reproducibility.
- Excluded `appointment_id` because it is an identifier.
- Excluded `fee_charged` because it is not available at booking time and could introduce data leakage.
- Excluded `no_show` from the input features because it is the prediction target.
- Used numeric and categorical preprocessing within a scikit-learn pipeline.
- Applied median imputation and standard scaling to numeric features.
- Applied most-frequent imputation and one-hot encoding to categorical features.

### Model Selection and Tuning

Compared a Dummy Classifier, Logistic Regression, balanced Logistic Regression, Decision Tree, and Random Forest using five-fold stratified cross-validation on the training set.

Hyperparameter tuning used GridSearchCV with six combinations of Logistic Regression parameters:

- `C`: 0.1, 1.0, 10.0
- `class_weight`: None, balanced

The selected configuration was:

- `C = 0.1`
- `class_weight = balanced`

The best cross-validation ROC-AUC was 0.7275.

The improvement over the initial Logistic Regression cross-validation ROC-AUC of 0.7273 was only 0.0002, which is smaller than the observed cross-validation standard deviation. Therefore, the improvement was treated as negligible rather than a meaningful performance gain.

### Threshold Selection

The classification threshold was selected using out-of-fold probabilities from the training data.

The selected threshold was **0.48**.

The test set was kept separate from model selection and threshold selection and was used for final evaluation.

### Cost Calculation

The clinic's estimated outreach costs were defined as:

- $8 for every patient flagged for contact.
- $35 for every no-show that was not flagged.

The model was compared against two baseline strategies: contacting nobody and contacting everybody.

## Model Comparison

Five-fold stratified cross-validation was performed on the training data. The detailed mean and standard deviation for accuracy, precision, recall, F1, and ROC-AUC are available