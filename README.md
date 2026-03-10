# Lake Mendota Ice Prediction – Polynomial Regression
This project demonstrates **training a polynomial regression model** on historical ice coverage data for **Lake Mendota** from 1855–56 to 2022–23. The model predicts future ice days, visualizes historical trends, and analyzes the rate of change in ice coverage over time.

## Project Overview
The dataset (`ice_data.csv`) contains:
- **Year (`x`)** – starting year of winter (e.g., 1855 for 1855–56)  
- **Number of ice days (`y`)** – total days the lake was frozen that year  

The project tasks include:
- Visualizing historical ice coverage trends  
- Normalizing data and constructing polynomial features  
- Training a polynomial regression model using **closed-form least squares**  
- Training the model using **gradient descent** and tracking convergence  
- Predicting the number of ice days for **2023–24**  
- Estimating and interpreting the local rate of change in ice coverage  

## Features Implemented
- **Data Visualization:** Generates `data_plot.jpg` (year vs. ice days)  
- **Feature Engineering:** Min-max normalization and polynomial feature augmentation  
- **Closed-Form Solution:** Computes optimal polynomial coefficients using the normal equation  
- **Gradient Descent:** Vectorized implementation with adjustable learning rate and iterations; generates `loss_plot.jpg`  
- **Prediction & Analysis:** Predicts ice days for 2023–24 and computes local rate of change  

## Python Libraries Used 
- **NumPy** – numerical computations, linear algebra, and arrays  
- **Pandas** – data loading, cleaning, and manipulation  
- **Matplotlib** – plotting and data visualization  

## Usage
Run the program as follows:
```bash
python3 lake_mendota_regression.py <filename.csv> <degree> <learning_rate> <iterations>
