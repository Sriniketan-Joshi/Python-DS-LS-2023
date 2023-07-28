import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Logistic_Regression:
    def _init_(self, learning_rate=0.01, num_iterations=10000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
        
    def split(self, data_frame):
        # Shuffle the data
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)
        
        # Split into train (80%) and test (20%)
        train_size = int(0.8 * len(data_frame))
        train_data = data_frame.iloc[:train_size]
        test_data = data_frame.iloc[train_size:]
        
        return train_data, test_data
    
    def fit(self, train_df, test_df):
        num_features_train = train_df.shape[1]
        num_features_test = test_df.shape[1]
        
        x_train = train_df.iloc[:, :num_features_train-1]
        y_train = train_df.iloc[:, -1]
        x_test = test_df.iloc[:, :num_features_test-1]
        y_test = test_df.iloc[:, -1]
        
        return x_train, y_train, x_test, y_test
    
    
    def GD(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)  # Initialize parameters to zeros
        self.bias = 0
        dj_dw = np.zeros(n)
        dj_db = 0.
        
               
        for _ in range(10000):
            z = np.dot(X, self.weights) + self.bias
            
            y_hat = self.sigmoid(z)
            
            dj_dw = (1/m) * (np.dot(X.T, (y_hat - y)))
            dj_db = (1/m) * (np.sum(y_hat - y))

            # Update Parameters using w, b, alpha and gradient
            self.weights -= 0.01 * dj_dw              
            self.bias -= 0.01 * dj_db              

        w_in = self.weights
        b_in = self.bias
        return w_in, b_in

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_hat = self.sigmoid(z)
        P = (y_hat > 0.5).astype(int)
        return P