import numpy as np


class IIFBaseClassifier:

    def __init__(self, C=0.01, kernel='linear', gamma=0.1, B=50, random_state=42):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.B = B
        self.rng = np.random.RandomState(random_state)

        self.t = 0
        self.n_classifiers = 0
        self.w = []
        self.SV = []
        self.alpha = []

    def _compute_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            X1 = np.asarray(X1, dtype=np.float64)
            X2 = np.asarray(X2, dtype=np.float64)
            dist_sq = (
                np.sum(X1 ** 2, axis=1)[:, np.newaxis]
                + np.sum(X2 ** 2, axis=1)[np.newaxis, :]
                - 2 * (X1 @ X2.T)
            )
            return np.exp(-self.gamma * np.maximum(dist_sq, 0.0))
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _ensure_capacity(self, X_t_parts):
        while self.n_classifiers < self.t:
            m_new = X_t_parts[self.n_classifiers].shape[1]
            if self.kernel == 'linear':
                self.w.append(np.zeros(m_new, dtype=np.float64))
            else:
                self.SV.append(np.empty((0, m_new), dtype=np.float64))
                self.alpha.append(np.array([], dtype=np.float64))
            self.n_classifiers += 1

    def _n_classifiers(self):
        return self.n_classifiers

    def fit_micro_block(self, X_t_parts, Y_t):
        X_t_parts = [np.asarray(p, dtype=np.float64) for p in X_t_parts]
        Y_t = np.asarray(Y_t, dtype=np.float64).ravel()

        self.t = len(X_t_parts)
        n_t = len(Y_t)
        if n_t == 0:
            return

        self._ensure_capacity(X_t_parts)

        l_t = np.zeros(self.t, dtype=np.float64)
        for j in range(self.t):
            X_tj = X_t_parts[j]
            if self.kernel == 'linear':
                f_j = X_tj @ self.w[j]
            else:
                if len(self.SV[j]) == 0:
                    f_j = np.zeros(n_t)
                else:
                    K_j = self._compute_kernel(X_tj, self.SV[j])
                    f_j = K_j @ self.alpha[j]

            margin = 1.0 - Y_t * f_j
            l_t[j] = max(0.0, float(np.mean(margin)))

        norm_l2 = float(np.linalg.norm(l_t, 2))
        if norm_l2 > 1e-12:
            tau_t = (self.C * l_t) / norm_l2
        else:
            tau_t = np.zeros(self.t, dtype=np.float64)

        for j in range(self.t):
            if l_t[j] > 0 and tau_t[j] > 0:
                X_tj = X_t_parts[j]

                if self.kernel == 'linear':
                    grad = np.sum(Y_t[:, np.newaxis] * X_tj, axis=0) / n_t
                    self.w[j] += tau_t[j] * grad
                else:
                    new_alpha = (tau_t[j] / n_t) * Y_t
                    mask = (new_alpha != 0)
                    if mask.any():
                        X_new_sv = X_tj[mask]
                        alpha_new_sv = new_alpha[mask]
                        if len(self.SV[j]) == 0:
                            self.SV[j] = X_new_sv.copy()
                            self.alpha[j] = alpha_new_sv
                        else:
                            self.SV[j] = np.vstack([self.SV[j], X_new_sv])
                            self.alpha[j] = np.concatenate([self.alpha[j], alpha_new_sv])

                    if len(self.SV[j]) > self.B:
                        idx = self.rng.choice(len(self.SV[j]),
                                              size=self.B,
                                              replace=False)
                        self.SV[j] = self.SV[j][idx]
                        self.alpha[j] = self.alpha[j][idx]

    def predict_parts(self, X_parts):
        X_parts = [np.asarray(p, dtype=np.float64) for p in X_parts]
        n_samples = X_parts[0].shape[0]
        scores = np.zeros(n_samples, dtype=np.float64)

        n_clf = self._n_classifiers()
        for j in range(min(len(X_parts), n_clf)):
            X_tj = X_parts[j]
            if self.kernel == 'linear':
                scores += X_tj @ self.w[j]
            else:
                if len(self.SV[j]) > 0:
                    K_j = self._compute_kernel(X_tj, self.SV[j])
                    scores += K_j @ self.alpha[j]

        preds = np.sign(scores)
        preds[preds == 0] = 1
        return preds, scores


class IIFStreamClassifier:

    def __init__(self, buffer_size=32, C=0.01, kernel='linear',
                 gamma=0.1, B=50, random_state=42):
        self.buffer_size = buffer_size
        self.iif_engine = IIFBaseClassifier(
            C=C, kernel=kernel, gamma=gamma, B=B, random_state=random_state
        )
        self.buffer_X = []
        self.buffer_y = []

    @staticmethod
    def _align_vector(x, target_dim):
        x = np.asarray(x, dtype=np.float64).ravel()
        d = x.shape[0]
        if d < target_dim:
            x = np.pad(x, (0, target_dim - d), mode='constant')
        elif d > target_dim:
            x = x[:target_dim]
        return x

    def _split_feature_vector(self, x, feature_dims):
        parts = []
        col_start = 0
        for d in feature_dims:
            col_end = col_start + d
            if x.ndim == 1:
                parts.append(x[col_start:col_end].reshape(1, -1))
            else:
                parts.append(x[:, col_start:col_end])
            col_start = col_end
        return parts

    def step(self, x_sample, y_sample, feature_dims):
        target_dim = int(sum(feature_dims))
        x_aligned = self._align_vector(x_sample, target_dim)
        self.buffer_X.append(x_aligned)
        self.buffer_y.append(y_sample)

        if len(self.buffer_X) >= self.buffer_size:
            self.flush_buffer(feature_dims)

    def flush_buffer(self, feature_dims):
        if not self.buffer_X:
            return

        target_dim = int(sum(feature_dims))
        X_aligned = [self._align_vector(xi, target_dim) for xi in self.buffer_X]
        X_batch = np.stack(X_aligned, axis=0)
        Y_batch = np.asarray(self.buffer_y, dtype=np.float64).ravel()

        X_parts = self._split_feature_vector(X_batch, feature_dims)
        self.iif_engine.fit_micro_block(X_parts, Y_batch)

        self.buffer_X.clear()
        self.buffer_y.clear()

    def predict_one(self, x_sample, feature_dims):
        target_dim = int(sum(feature_dims))
        x_aligned = self._align_vector(x_sample, target_dim)
        X_parts = self._split_feature_vector(x_aligned, feature_dims)
        preds, scores = self.iif_engine.predict_parts(X_parts)
        return preds[0], scores[0]

    def predict_batch(self, X_batch, feature_dims):
        target_dim = int(sum(feature_dims))
        X_batch = np.asarray(X_batch, dtype=np.float64)
        if X_batch.ndim == 2 and X_batch.shape[1] != target_dim:
            if X_batch.shape[1] < target_dim:
                pad = np.zeros((X_batch.shape[0], target_dim - X_batch.shape[1]))
                X_batch = np.hstack([X_batch, pad])
            else:
                X_batch = X_batch[:, :target_dim]
        X_parts = self._split_feature_vector(X_batch, feature_dims)
        return self.iif_engine.predict_parts(X_parts)

    def predict(self, x, feature_dims=None):
        if feature_dims is None:
            feature_dims = [len(x)]
        pred_label, _ = self.predict_one(x, feature_dims)
        return pred_label

    def partial_fit(self, x, y, feature_dims=None):
        if feature_dims is None:
            feature_dims = [len(x)]
        y_iif = 1 if int(y) == 1 else -1
        self.step(x_sample=x, y_sample=y_iif, feature_dims=feature_dims)
