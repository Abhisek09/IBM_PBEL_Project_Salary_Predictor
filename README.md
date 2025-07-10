
# Salary Prediction using Ensemble Learning

A machine learning app that predicts salaries based on user input or uploaded datasets using regression models.

Live App: https://ibmpbelprojectsalarypredictor.streamlit.app/



## Features


- Predict salary for individual entries via a simple form
- Bulk predictions via CSV upload
- Compare predicted salary with actual salary using MAE, RMSE, and R²
- Uses an ensemble of regression models with XGBoost as the best performer
- Integrated model evaluation for performance testing
- Deployed with Streamlit for an interactive web experience
## Installation & Usage

Follow these steps to run the app locally:

Clone the Repository
```bash
  git clone https://github.com/Abhisek09/IBM_PBEL_Project_Salary_Predictor.git
```


Install Dependencies
```bash
  pip install -r requirements.txt
```

Run the Streamlit App
```bash
  streamlit run SPTM21.py
```

## Features


- Predict salary for individual entries via a simple form
- Bulk predictions via CSV upload
- Compare predicted salary with actual salary using MAE, RMSE, and R²
- Uses an ensemble of regression models with XGBoost as the best performer
- Integrated model evaluation for performance testing
- Deployed with Streamlit for an interactive web experience
## Tech Stack

Language: Python

Libraries: Pandas, Scikit-learn, XGBoost, Joblib

ML Model: XGBoost Regressor (Best performer among ensemble models)

Deployment: Streamlit

## Screenshots and Guide

# Individual Entry
![image](https://github.com/user-attachments/assets/349b57d8-c8ab-4033-a757-129a0465943a)
- Select the Individual Entry and then proceed filling the required data
- once filled click on the predict to generate the predicted salary


# Bulk Entry
![image](https://github.com/user-attachments/assets/bc2a503f-e283-4eb4-b7c1-9f931e6e12d1)
- Select the Bulk entry, to upload the data in CSV/Excel format.
- Once the file is uploaded it will generate the output, we will be able to view it and download it as well.


## Contact

Abhisek Jena

Email - abhisek.jena48@gmail.com
