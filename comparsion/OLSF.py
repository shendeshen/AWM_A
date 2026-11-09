import numpy as np
import random

class OLSF:
    def __init__(self, variant='I', C=1.0, lamda=30.0, B=1.0, random_state=None):
        self.variant = variant
        self.C = C
        self.lamda = lamda
        self.B = B
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state) if random_state is not None else None

        self.w = np.array([], dtype=float)


    def _project_features(self, x, target_dim):
        x = np.asarray(x, dtype=float)
        if len(x) >= target_dim:
            return x[:target_dim]
        return np.pad(x, (0, target_dim - len(x)), mode='constant')

    def _split_features(self, x):
        x = np.asarray(x, dtype=float)
        d_curr = len(x)
        d_prev = len(self.w)
        tilde_x = self._project_features(x, d_prev)
        if d_curr > d_prev:
            hat_x = x[d_prev:]
        else:
            hat_x = np.array([], dtype=float)
        return tilde_x, hat_x

    def _hinge_loss(self, x, y):
        if len(self.w) == 0:
            return 1.0
        margin = y * np.dot(self.w, x)
        return max(0.0, 1.0 - margin)

    def _compute_tau(self, loss, x_norm_sq):
        if x_norm_sq <= 1e-12:
            return 0.0
        if self.variant == 'base':
            return loss / x_norm_sq
        elif self.variant == 'I':
            return min(self.C, loss / x_norm_sq)
        elif self.variant == 'II':
            return loss / (x_norm_sq + 1.0 / (2.0 * self.C))
        else:
            raise ValueError("variant must be 'base', 'I', or 'II'")

    def _project_L1_ball(self, w):
        if len(w) == 0:
            return w
        l1_norm = np.linalg.norm(w, ord=1)
        if l1_norm <= self.lamda:
            return w
        return (self.lamda / l1_norm) * w

    def _truncate_sparse(self, w):
        if self.B >= 1.0:
            return w
        d = len(w)
        if d == 0:
            return w
        keep_num = max(1, int(np.floor(self.B * d)))
        nonzero_num = np.count_nonzero(w)
        if nonzero_num <= keep_num:
            return w
        top_idx = np.argsort(np.abs(w))[-keep_num:]
        sparse_w = np.zeros_like(w)
        sparse_w[top_idx] = w[top_idx]
        return sparse_w

    def decision_function(self, x):
        if len(self.w) == 0:
            return 0.0
        x_proj = self._project_features(x, len(self.w))
        return np.dot(self.w, x_proj)

    def predict(self, x):
        if len(self.w) == 0:
            return 0
        score = self.decision_function(x)
        return 1 if score >= 0 else 0

    def partial_fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y_signed = 1 if y == 1 else -1
        d_curr = len(x)
        d_prev = len(self.w)
        if d_prev == 0:
            self.w = np.zeros(d_curr, dtype=float)

        tilde_x, hat_x = self._split_features(x)
        loss = self._hinge_loss(tilde_x, y_signed)

        if d_curr > len(self.w):
            expand_dim = d_curr - len(self.w)
            self.w = np.pad(self.w, (0, expand_dim), mode='constant')

        if loss <= 1e-12:
            return self

        x_norm_sq = np.dot(tilde_x, tilde_x) + np.dot(hat_x, hat_x) + 1e-12

        tau = self._compute_tau(loss, x_norm_sq)


        self.w[:len(tilde_x)] += tau * y_signed * tilde_x

        if len(hat_x) > 0:
            start = len(tilde_x)
            self.w[start:start+len(hat_x)] = tau * y_signed * hat_x

        self.w = self._project_L1_ball(self.w)
        self.w = self._truncate_sparse(self.w)

        return self

    def reset(self):
        self.w = np.array([], dtype=float)

    def get_weights(self):
        return self.w.copy()
class BaggingOLSF:

    def __init__(self, n_estimators=5, variant='I', C=1.0, lamda=30.0, B=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.rng = np.random.RandomState(random_state)
        self.estimators = []
        for i in range(n_estimators):
            child_seed = self.rng.randint(2**31) if random_state is not None else None
            est = OLSF(variant=variant, C=C, lamda=lamda, B=B, random_state=child_seed)
            self.estimators.append(est)

    def partial_fit(self, x, y):

        for est in self.estimators:
            k = self.rng.poisson(1)
            for _ in range(k):
                est.partial_fit(x, y)

    def predict(self, x):
        votes = [est.predict(x) for est in self.estimators]
        return 1 if sum(votes) > self.n_estimators // 2 else 0

    def predict_proba(self, x):
        votes = [est.predict(x) for est in self.estimators]
        return sum(votes) / self.n_estimators
