import numpy as np

def distanceFunc(metric_type, vec1, vec2):
    """
    Computes the distance between two d-dimension vectors. 
    
    Please DO NOT use Numpy's norm function when implementing this function. 
    
    Args:
        metric_type (str): Metric: L1, L2, or L-inf
        vec1 ((d,) np.ndarray): d-dim vector
        vec2 ((d,)) np.ndarray): d-dim vector
    
    Returns:
        distance (float): distance between the two vectors
    """

    diff = vec1 - vec2
    if metric_type == "L1":
        distance = np.sum(np.abs(diff), axis=-1)

    if metric_type == "L2":
        distance = np.sqrt(np.sum(np.square(diff), axis=-1))
        
    if metric_type == "L-inf":
        distance = np.max(np.abs(diff), axis=-1)
        
    return distance


def computeDistancesNeighbors(K, metric_type, X_train, y_train, sample):
    """
    Compute the distances between every datapoint in the train_data and the 
    given sample. Then, find the k-nearest neighbors.
    
    Return a numpy array of the label of the k-nearest neighbors.
    
    Args:
        K (int): K-value
        metric_type (str): metric type
        X_train ((n,p) np.ndarray): Training data with n samples and p features
        y_train : Training labels
        sample ((p,) np.ndarray): Single sample whose distance is to computed with every entry in the dataset
        
    Returns:
        neighbors (list): K-nearest neighbors' labels
    """

    # You will also call the function "distanceFunc" here
    # Complete this function
    dist = distanceFunc(metric_type, X_train, sample[np.newaxis, :])
    neighbors = y_train[np.argsort(dist)[:K]]
    return neighbors


def Majority(neighbors):
    """
    Performs majority voting and returns the predicted value for the test sample.
    
    Since we're performing binary classification the possible values are [0,1].
    
    Args:
        neighbors (list): K-nearest neighbors' labels
        
    Returns:
        predicted_value (int): predicted label for the given sample
    """
    
    # Performs majority voting
    # Complete this function
    count = np.bincount(neighbors)
    predicted_value = np.argmax(count)
    return predicted_value


def KNN(K, metric_type, X_train, y_train, X_val):
    """
    Returns the predicted values for the entire validation or test set.
    
    Please DO NOT use Scikit's KNN model when implementing this function. 

    Args:
        K (int): K-value
        metric_type (str): metric type
        X_train ((n,p) np.ndarray): Training data with n samples and p features
        y_train : Training labels
        X_val ((n, p) np.ndarray): Validation or test data
        
    Returns:
        predicted_values (list): output for every entry in validation/test dataset 
    """
    
    # Complete this function
    # Loop through the val_data or the test_data (as required)
    # and compute the output for every entry in that dataset  
    # You will also call the function "Majority" here
    predictions = Majority(computeDistancesNeighbors(K, metric_type, X_train, y_train, X_val))
    return predictions

def main():
    n = int(input())
    x_train = np.empty(shape=(n, 30), dtype=float)
    y_train = np.empty(shape=(n, ), dtype=int)
    for i in range(n):
        data = list(map(float, input().split()))
        y_train[i] = int(data[0])
        x_train[i] = data[1:]

    n = int(input())
    samples = np.empty(shape=(n, 30), dtype=float)
    K = np.empty(shape=(n, ), dtype=int)
    for i in range(n):
        data = list(map(float, input().split()))
        K[i] = int(data[0])
        samples[i] = data[1:]
    
    pred = [KNN(k, "L2", x_train, y_train, sample) for k, sample in zip(K, samples)]
    for p in pred:
        print(p)

if __name__ == '__main__':
    main()
    