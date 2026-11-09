import numpy as np
from numpy import random
from scipy.linalg import pinv, inv


class LiteRVFL_Online:


    def __init__(self,
                 n_hidden=100,
                 reg=1e-6,
                 forgetting_rate=0.995,
                 activation='sigmoid',
                 n_classes=6,
                 random_state=None):

        self.n_hidden = n_hidden
        self.reg = reg
        self.lambda_ = forgetting_rate
        self.activation = activation
        self.n_classes = n_classes
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.W = None
        self.b = None
        self.beta = None
        self.P = None
        self._mean = None
        self._std = None
        self.n_samples = 0



    def _initialize_random_weights(self, n_features):
        self.W = 2 * random.random((self.n_hidden, n_features)) - 1
        self.b = 2 * random.random((self.n_hidden, 1)) - 1

        self.W = self._orthogonalize(self.W)

    def _orthogonalize(self, W):
        W = W.T.copy()
        for i in range(W.shape[1]):
            for j in range(i):
                W[:, i] -= np.dot(W[:, j], W[:, i]) * W[:, j]
            W[:, i] /= (np.linalg.norm(W[:, i]) + 1e-8)
        return W.T

    def _activation(self, x):
        if self.activation == 'sigmoid':
            return 1.0 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            return x

    def _random_mapping(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        H = X @ self.W.T + self.b.T
        return self._activation(H)

    def _standardize(self, X, update=True):
        X = np.asarray(X).reshape(-1, X.shape[-1]) if X.ndim == 1 else np.asarray(X)

        if self._mean is None:
            self._mean = np.zeros(X.shape[1])
            self._std = np.ones(X.shape[1])

        if update:
            for x in X:
                self.n_samples += 1
                delta = x - self._mean
                self._mean += delta / self.n_samples
                delta2 = x - self._mean
                if self.n_samples > 1:
                    self._std = np.sqrt(((self.n_samples - 1) * self._std ** 2 + delta * delta2) / self.n_samples)

        return (X - self._mean) / (self._std + 1e-8)

    def partial_fit(self, X, y):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            y = np.array([y])

        batch_size = X.shape[0]

        if self.W is None:
            self._initialize_random_weights(X.shape[1])

        X_scaled = self._standardize(X, update=True)
        H = self._random_mapping(X_scaled)

        Y = np.zeros((batch_size, self.n_classes))
        Y[np.arange(batch_size), y] = 1.0

        if self.beta is None:
            HTH = H.T @ H + self.reg * np.eye(self.n_hidden)
            self.beta = pinv(HTH) @ (H.T @ Y)
            self.P = inv(HTH + 1e-8 * np.eye(self.n_hidden))
            return self

        for i in range(batch_size):
            h = H[i:i + 1, :].T
            y_i = Y[i:i + 1, :].T

            denom = self.lambda_ + (h.T @ self.P @ h).item()
            K = (self.P @ h) / denom
            error = y_i - (self.beta.T @ h)
            self.beta += K @ error.T
            self.P = (self.P - K @ h.T @ self.P) / self.lambda_

        return self

    def predict(self, X):

        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):

        if self.beta is None:
            n = len(X) if hasattr(X, '__len__') else 1
            return np.ones((n, self.n_classes)) / self.n_classes

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = (X - self._mean) / (self._std + 1e-8)
        H = self._random_mapping(X_scaled)

        logits = H @ self.beta

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        proba = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-8)
        return proba

    def reset(self):

        self.W = None
        self.b = None
        self.beta = None
        self.P = None
        self._mean = None
        self._std = None
        self.n_samples = 0