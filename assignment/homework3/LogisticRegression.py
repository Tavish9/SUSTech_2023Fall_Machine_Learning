import numpy as np

class LogisticRegression():
    def __init__(self, n_features, n_classes, max_epoch, lr) -> None:
        self.w = np.zeros((n_features + 1, n_classes))
        self.max_epoch = max_epoch
        self.lr = lr

    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

        logits = np.dot(X, self.w)
        probabilities = self._softmax(logits)

        return np.argmax(probabilities, axis=1)

    def fit(self, X, y):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        
        for _ in range(self.max_epoch):
            logits = np.dot(X, self.w)
            probabilities = self._softmax(logits)

            gradient = np.dot(X.T, probabilities - y) / X.shape[0]

            self.w -= self.lr * gradient

    def _softmax(self, X):
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
def main():
    INPUT = input().split()
    N, D, C, E = list(map(int, INPUT[:4]))
    lr = float(INPUT[-1])
    x_train = np.empty(shape=(N, D), dtype=float)
    y_train = np.empty(shape=(N, C), dtype=int)
    for i in range(N):
        data = list(map(float, input().split()))
        x_train[i] = data

    for i in range(N):
        data = list(map(int, input().split()))
        y_train[i] = data

    model = LogisticRegression(D, C, E, lr)
    model.fit(x_train, y_train)
    W = model.w.reshape(-1)
    for i in range(W.shape[0]):
        print("%.3f" % (W[i]))

if __name__ == '__main__':
    main()