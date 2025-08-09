# Streamlit Cancer Prediction App

## Overview
This project is a web application built with Streamlit that predicts the likelihood of breast cancer based on input features. It uses a machine learning model trained on a breast cancer dataset. The app provides an interactive interface for users to input data and get instant predictions.

## Features
- Predicts breast cancer (Malignant/Benign) using a trained ML model
- User-friendly Streamlit interface
- Theme toggle (Light/Dark mode)
- Data preview and summary

## Project Structure
```
├── app/
│   ├── assets/
│   │   └── styles.css
│   └── main.py
├── data/
│   └── data.csv
├── model/
│   ├── main.py
│   ├── model.pkl
│   └── scaler.pkl
├── requirements.txt
└── README.md
```

## Setup Instructions
1. **Clone the repository:**
	```bash
	git clone <repo-url>
	cd STREAMLIT-CANCER-APP
	```

2. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```

3. **Run the app:**
	```bash
	streamlit run app/main.py
	```

## Usage
1. Open the app in your browser (usually at http://localhost:8501).
2. Use the sidebar to toggle between Light and Dark themes.
3. Enter the required features for prediction.
4. Click the Predict button to see the result.

## Data
The dataset used is located at `data/data.csv` and is preprocessed in the app.

## Model
The trained model (`model.pkl`) and scaler (`scaler.pkl`) are stored in the `model/` directory.

## Customization
- You can modify the CSS in `app/assets/styles.css` for further UI customization.
- To retrain the model, update `model/main.py` as needed.

## Credits
- Built with [Streamlit](https://streamlit.io/)
- Dataset: [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

## License
This project is for educational purposes.
