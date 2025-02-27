import numpy as np


def normalized_mae(labels, predictions):
    # Convert to numpy arrays for easy vectorized computation
    labels = np.array(labels)
    predictions = np.array(predictions)

    # Calculate the absolute error, normalized by the true value
    error = np.abs(predictions - labels) / np.maximum(1, labels)
    
    # Calculate the mean of the normalized errors
    normalized_error = np.mean(error)
    
    return normalized_error