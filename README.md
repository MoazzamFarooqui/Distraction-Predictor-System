# Distraction Predictor System

A machine learning project that predicts whether a person is focused or distracted based on their daily habits. The model is trained on lifestyle data and deployed using Streamlit, allowing users to evaluate their focus status through an interactive web application.

---

## Badges

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge\&logo=scikitlearn\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Project-success?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

---

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Dataset](#dataset)
* [Technologies Used](#technologies-used)
* [Machine Learning Workflow](#machine-learning-workflow)
* [Installation](#installation)
* [Usage](#usage)
* [Screenshots](#screenshots)

---

# Project Overview

Maintaining focus and productivity depends on several daily habits. Factors such as study hours, sleep duration, and phone usage can significantly influence a person's ability to stay focused.

This project uses a Decision Tree Classifier to learn these patterns from historical data and predict whether a person is likely to be focused or distracted. The trained model is integrated into a Streamlit web application where users can adjust their daily habits and receive an instant prediction.

---

# Features

* Predicts whether a user is Focused or Distracted
* Interactive web application built with Streamlit
* Clean and modern user interface
* Slider-based inputs for quick interaction
* Real-time predictions using a trained Machine Learning model
* Responsive prediction results with custom styling

---

# Dataset

The dataset contains information about daily habits, including:

* Study Hours
* Sleep Hours
* Phone Usage
* Focus (Target Variable)

These features are used to train the classification model and generate focus predictions.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib
* Jupyter Notebook

---

# Machine Learning Workflow

1. Load the dataset.
2. Preprocess the data.
3. Split the dataset into training and testing sets.
4. Train a Decision Tree Classifier.
5. Evaluate the model's performance.
6. Save the trained model using Joblib.
7. Build an interactive Streamlit application.
8. Generate predictions based on user input.

---

# Installation

### Clone the repository

```bash
git clone https://github.com/MoazzamFarooqui/Distraction-Predictor-System.git
```

### Navigate to the project directory

```bash
cd Distraction-Predictor-System
```

### Install the required packages

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

# Usage

1. Launch the Streamlit application.
2. Adjust the Study Hours, Sleep Hours, and Phone Usage sliders.
3. Click **Analyze My Focus**.
4. View your predicted focus status.

---

