import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Read in inputs
    filename = sys.argv[1]
    degree = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    iterations = int(sys.argv[4])
    
    # Data visualization
    
    # plot the number of frozen days vs the year from the data set in ice_data.csv
    data = pd.read_csv(filename) # read the data from the csv file
    plt.figure() # start a new figure
    plt.scatter(data['year'], data['days'])
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.title('Frozen Days vs Year')
    plt.savefig('data_plot.jpg') # save the plot as a jpg file
    plt.close() # close the figure
    
    # Feature engineering
    
    # calculate xmin and xmax for the year data
    xmin = data['year'].min()
    xmax = data['year'].max()
    
    # calculate x_i = (x_i - x_min) / (x_max - x_min) for each year data point
    data['year_normalized'] = (data['year'] - xmin) / (xmax - xmin)
    
    # create a feature vector for each data point using the year and the degree of the polynomial
    feature_list = []
    # for every power, calculate all x_i^power and add it to the feature list as a new row 
    for i in range(1, degree + 1):
        # (degree + 1)  * n matrix where degree + 1 is the number of data points and n is the number of features (degree + 1)
        feature_list.append(data['year_normalized'] ** i)
    # bias as last column
    feature_list.append(np.ones(len(data)))  
    # convert the feature list to a numpy array and transpose it to n*m get the feature vector
    # feature vector has every row as a data point and every column as a feature (degree + 1)
    X_tilde = np.array(feature_list).T 
    
    # lift data into a higher-dimensional feature space so a linear model can express nonlinear behavior.
    print("Feature matrix:")
    print(X_tilde)
    
    # Closed-form linear regression solution
        
    # calculate the weights and bias using the closed-form solution by taking the derivative of the MSE loss with respect to w and b and setting it to zero
    # this will find the global minimum of the MSE loss function and will not require us to iterate over the data points like in gradient descent
    theta_closed_form = np.linalg.lstsq(X_tilde, data['days'].values, rcond=None)[0]
    
    # process is computationally expensive for large datasets because it involves computing the inverse of a matrix, which has a time complexity of O(n^3) where n is the number of features. 
    # For small datasets, it can be efficient and provides an exact solution, but for larger datasets, it may not be practical due to the computational cost and potential numerical instability when inverting matrices.    
    print("Closed-form parameters:")
    print(theta_closed_form)
    
    # Gradient descent training
        
    
    # initialize the parameter vector theta to zeros and bias to zero
    theta_gradient_descent = np.zeros(degree + 1) 
    
    # initialize a list to hold the mse_loss values for each iteration of gradient descent
    mse_loss_list = []
    
    print("Gradient descent parameter trace:")
    
        
    for t in range(iterations):
        
        # print out the current parameter vector theta once every 10 iterations
        if t % 10 == 0:
            print(theta_gradient_descent)
        
        # calculate the predicted values using the current parameter vector theta and the feature vector X_tilde
        # bias is included in feature vector last column and theta last element, so we can just do a dot product to get the predicted values
        y_pred = X_tilde.dot(theta_gradient_descent)
        y = data['days'].values
        
        # calculate the MSE loss for the current parameter vector theta and the predicted values y_pred
        mse_loss = (1/(2 * len(data)) * np.sum((y_pred - y) ** 2))
        mse_loss_list.append(mse_loss)
        
        # calculate the gradient of the MSE loss with respect to theta
        # gradient formula: ∇L(θ) = (1/n) * X̃^T * (X̃θ - Y)
        # mutliply by X_tilde.T to to know how much each feature contributes to the error
        # divide by the number of data points to get the average gradient
        
        # loss_gradient is a vector of partial derivatives of the MSE loss with respect to each element of theta
        # it tells us how much we need to change each element of theta to minimize the MSE loss
        loss_gradient = (1/len(data)) * X_tilde.T.dot(y_pred - y)
        
        # update theta so it moves in the direction of the negative gradient to minimize the MSE loss
        # learning_rate controls how big of a step we take in the direction of the negative gradient, if it's too large we might overshoot the minimum and if it's too small it will take a long time to converge
        # the minus sign indicates we are moving in the direction of the negative gradient
        theta_gradient_descent = theta_gradient_descent - learning_rate * loss_gradient
    
    
    print("Learning rate: " + str(learning_rate))
    print("Iterations: " + str(iterations))
    
    # describe your tuning process for finding the optimal learning rate and number of iterations in the gradient descent algorithm.
    print("Tuning notes: I started with a small learning rate of 0.01 but convergence was slow. I gradually increased the learning rate, testing 0.1, 0.3, and 0.5, while monitoring loss. Higher rates converged too quickly and made the curve less informative. I selected 0.7 as a balance between convergence speed and curve visibility, and used 100000 iterations to approach the closed-form loss without divergence.")
    
    # plot the MSE loss
    print("Saving loss plot...")
    plt.figure()
    plt.plot(mse_loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('MSE Loss vs Iteration')
    plt.yscale('log')
    plt.savefig('mse_loss_plot.jpg') # save the plot as a jpg file
    plt.close() # close the figure
        
    # Prediction for year 2023
    
    # make a prediction for the number of frozen days in the year 2023 using the parameter vector theta from the closed-form solution
    year_2023_normalized = (2023 - xmin) / (xmax - xmin)
    
    input_vector = []
    
    # create a feature vector for the year 2023 using the same process as before
    for i in range(1, degree + 1):
        input_vector.append(year_2023_normalized ** i)
    # add last column for bias term
    input_vector.append(1)
    # convert to numpy array for matrix multiplication
    input_vector = np.array(input_vector)
    
    # prediction is coefficient vector theta from closed-form solution dot product with the input feature vector for the year 2023
    y_pred_2023_closed_form = np.dot(theta_closed_form, input_vector)
    
    print("Predicted frozen days in 2023: " + str(y_pred_2023_closed_form))
    
    
    # Local rate of change near 2023
    
    # calculate the model's local rate of change at winter 2023-2024 
    # so calculate the derivative of the predicted number of frozen days with respect to the year at the year 2023
    rate_of_change = 0
    
    # use derivative of polynomial formula
    # theta is in [x^1, x^2, x^3, ..., bias], so the derivative with respect to year is the sum of i * theta[i] * (year_normalized)^(i-1) for i in range(1, degree + 1)
    for i in range(1, degree + 1):
        rate_of_change += i * theta_closed_form[i-1] * (year_2023_normalized ** (i-1))
    
    print("Rate of change at 2023: " + str(rate_of_change))
    print("Interpretation: A negative rate of change indicates the number of ice days is decreasing around 2023, suggesting a warming trend and reduced lake ice formation.")