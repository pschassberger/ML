# cited https://beckernick.github.io/logistic-regression-from-scratch/



import numpy as np
import matplotlib.pyplot as plt


# get data
# randomly generated set
np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [.75, 1]], num_observations)

simulated_seperable_features = np.vstack((x1, x2)). astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations), 
                            np.ones(num_observations)))
'''# vizualize
plt.figure(figsize=(12, 8))
plt.scatter(simulated_seperable_features[:, 0], simulated_seperable_features[:, 1], 
            c=simulated_labels, alpha=0.4)
plt.show()'''

# Sigmoid function / link function
# The link function provides the relationship between the linear predictor and the mean of the distribution function. 
# There are many commonly used link functions, and their choice is informed by several considerations.
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

# Log likleyhood
# The log-likelihood can be viewed as a sum over all the training data

def log_likleyhood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum(target * scores - np.log(1 + np.exp(scores)))
    return ll

# Build the Logistic Regression Function
def logistic_regression(features, target, num_steps, learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # update weights with gradient
        outpu_error_signal = target - predictions
        gradient = np.dot(features.T, outpu_error_signal)
        weights += learning_rate * gradient

        # print log-likleyhood
        '''if step % 10000 == 0:
            print(log_likleyhood(features, target, weights))'''

    return weights

weights = logistic_regression(simulated_seperable_features, simulated_labels, 
         num_steps=300000, learning_rate=5e-5, add_intercept=True)

# check accuracy 
data_with_intercept = np.hstack((np.ones((simulated_seperable_features.shape[0], 1)),
                        simulated_seperable_features))

final_scores = np.dot(data_with_intercept, weights)
preds = np.round(sigmoid(final_scores))

print("Accuracy: ", (preds == simulated_labels).sum().astype(float) / len(preds))