from typing import Tuple
import numpy as np
# import scipy.linalg 


class Model:
    # Modify your model, default is a linear regression model with random weights
    ID_DICT = {"NAME": "Qilin Yang", "BU_ID": "U73204239", "BU_EMAIL": "yangql15@bu.edu"}

    def __init__(self):
        self.theta = None

    def preprocess(self, X: np.array, y: np.array) -> Tuple[np.array, np.array]:
        ###############################################
        ####      add preprocessing code here      ####
        ###############################################

#         def feature_normalize(samples):

#             means = np.mean(samples,axis=0)
#             sample_norm = samples - means
#             std = np.std(sample_norm,axis=0)
#             sample_norm = np.divide(sample_norm,std, where=std!=0)
#             return means, std,sample_norm
        
        
#         def get_usv(sample_norm):
#             # Compute the covariance matrix
#             sigma = (sample_norm.T.dot(sample_norm))/len(sample_norm)
            
            
#             U,_,_ = np.linalg.svd(sigma)
#             return U
        
        
#         def project_data(samples, U, K):
    
#             # Reduced U is the first "K" columns in U
#             reduced_U = U[:,0:K]
#             z = samples.dot(reduced_U)
#             return z
        
        
#         def recover_data(Z, U, K):
#             return Z.dot(U[:,0:K].T)
        
#         means, stds, samples_norm = feature_normalize(X)

#         # Run SVD
#         U = get_usv(samples_norm)
#         z = project_data(samples_norm, U, 764)
#         X = z
#         print(X.shape)

        return X, y
    
    

    def train(self, X_train: np.array, y_train: np.array):
        
        
            
        """
        Train model with training data
        """
        ###############################################
        ####   initialize and train your model     ####
        ###############################################
        def loss(X,y,theta):
            h = X.dot(theta)
            
            diff = h-y
            
            n = X.shape[0]
            loss = np.sum(np.square(diff))/(2*n)
            
            return loss
        
        def loss_gradient(X,y, theta):
            pred = X.dot(theta)
            
            n = X.shape[0]
            loss_grad =X.T.dot(pred-y)/n
    
            return loss_grad
    

        X_train = np.vstack((np.ones((X_train.shape[0],)), X_train.T)).T
        self.theta = np.random.rand(X_train.shape[1], 1)

        
        theta_current = self.theta
        loss_values = []
        theta_values = []
        
        for i in range(5000):
            loss_value = loss(X_train, y_train, self.theta)
    
            theta_current -= 0.07*loss_gradient(X_train, y_train, self.theta)
            loss_values.append(loss_value)
            theta_values.append(theta_current)
        
    
        self.theta = theta_current
        return theta_current

    def predict(self, X_val: np.array) -> np.array:
        """
        Predict with model and given feature
        """
        ###############################################
        ####      add model prediction code here   ####
        ###############################################
        
        X_val = np.vstack((np.ones((X_val.shape[0],)), X_val.T)).T
        return np.dot(X_val, self.theta)
