import numpy as np
from collections import defaultdict


class RAIL:

    def __init__(self,
                 num_classes=2,
                 lr=0.01,
                 random_state=None,
                 n_gd_iter=10,
                 eps=1e-8,
                 alpha_range=(0.3, 1.8),
                 beta_range=(0.3, 1.8),
                 gamma_range=(0.8, 1.2)):

        self.num_classes = num_classes
        self.lr = lr
        self.random_state = random_state
        self.n_gd_iter = n_gd_iter
        self.eps = eps
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.gamma_range = gamma_range

        self.rng = np.random.RandomState(random_state)

        self.class_counts = np.zeros(num_classes, dtype=float)
        self.feature_means = defaultdict(lambda: np.zeros(num_classes))
        self.feature_vars = defaultdict(lambda: np.ones(num_classes))
        self.feature_class_counts = defaultdict(lambda: np.zeros(num_classes))

        self.n_bins = 10
        self.feature_bins = {}
        self.joint_xy = defaultdict(
            lambda: defaultdict(lambda: np.zeros(num_classes)))
        self.joint_xx = defaultdict(lambda: defaultdict(int))
        self.feature_counts = defaultdict(lambda: defaultdict(int))

        self.feature_seen = set()

        self.alpha = {}
        self.beta = {}
        self.gamma = float(self.rng.uniform(*gamma_range))

        self.D_cache = defaultdict(float)
        self.R_cache = defaultdict(float)

        self.recent_samples = []
        self.max_cache = 200

        self.n_samples = 0
        self.is_fitted = False
        self.loss_history = []

    def _init_feature_weights(self, fid):
        self.alpha[fid] = float(self.rng.uniform(*self.alpha_range))
        self.beta[fid] = float(self.rng.uniform(*self.beta_range))

    def _update_bins(self, fid, val):
        if fid not in self.feature_bins:
            self.feature_bins[fid] = [val - 1.0, val + 1.0]
            return
        edges = self.feature_bins[fid]
        if val < edges[0]:
            edges[0] = val
        if val > edges[1]:
            edges[1] = val

    def _get_bin_idx(self, fid, val):
        lo, hi = self.feature_bins[fid]
        if hi - lo < self.eps:
            return 0
        idx = int((val - lo) / (hi - lo) * (self.n_bins - 1))
        return max(0, min(self.n_bins - 1, idx))

    def partial_fit(self, X, y):
        if isinstance(X, (list, tuple)):
            X = np.array(X, dtype=float)
        if X.ndim > 1:
            X = X.flatten()
        y = int(y)
        if y < 0 or y >= self.num_classes:
            return

        feat_dict = {i: float(X[i]) for i in range(len(X))
                     if not np.isnan(X[i])}

        self.class_counts[y] += 1
        self.n_samples += 1
        cur_bins = {}

        for fid, val in feat_dict.items():
            if fid not in self.feature_seen:
                self.feature_seen.add(fid)
                self._init_feature_weights(fid)

            self._update_bins(fid, val)

            self.feature_class_counts[fid][y] += 1
            cnt = self.feature_class_counts[fid][y]
            old_mean = self.feature_means[fid][y]
            new_mean = old_mean + (val - old_mean) / cnt
            self.feature_means[fid][y] = new_mean
            if cnt > 1:
                old_var = self.feature_vars[fid][y]
                self.feature_vars[fid][y] = old_var + (
                    (val - old_mean) * (val - new_mean) - old_var) / cnt

            bin_idx = self._get_bin_idx(fid, val)
            cur_bins[fid] = bin_idx
            self.feature_counts[fid][bin_idx] += 1
            self.joint_xy[fid][bin_idx][y] += 1

        fids = list(cur_bins.keys())
        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                fi, fj = fids[i], fids[j]
                bi, bj = cur_bins[fi], cur_bins[fj]
                self.joint_xx[fi][(fj, bi, bj)] += 1
                self.joint_xx[fj][(fi, bj, bi)] += 1

        self.recent_samples.append((feat_dict.copy(), y))
        if len(self.recent_samples) > self.max_cache:
            self.recent_samples.pop(0)

        self.is_fitted = True

        if self.n_samples % 50 == 0:
            self._update_D_R()
            self._optimize_coefficients()

    def _mutual_info_xy(self, fid):
        total = self.n_samples
        if total == 0:
            return 0.0
        mi = 0.0
        for bin_idx, cnt_x in self.feature_counts[fid].items():
            if cnt_x == 0:
                continue
            px = cnt_x / total
            for y in range(self.num_classes):
                cnt_xy = self.joint_xy[fid][bin_idx][y]
                if cnt_xy == 0:
                    continue
                pxy = cnt_xy / total
                py = self.class_counts[y] / total
                mi += pxy * np.log((pxy + self.eps) / (px * py + self.eps))
        return max(0.0, mi)

    def _mutual_info_xx(self, fi, fj):
        total = self.n_samples
        if total == 0:
            return 0.0
        mi = 0.0
        for (fj_key, bi, bj), cnt in self.joint_xx[fi].items():
            if fj_key != fj or cnt == 0:
                continue
            pxy = cnt / total
            pi = self.feature_counts[fi].get(bi, 0) / total
            pj = self.feature_counts[fj].get(bj, 0) / total
            if pi > 0 and pj > 0:
                mi += pxy * np.log((pxy + self.eps) / (pi * pj + self.eps))
        return max(0.0, mi)

    def _update_D_R(self):
        fids = list(self.feature_seen)
        n = len(fids)
        if n == 0:
            return

        I_xy = {fid: self._mutual_info_xy(fid) for fid in fids}
        avg_I_xy = np.mean(list(I_xy.values())) + self.eps

        I_xx = {}
        pair_list = []
        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = fids[i], fids[j]
                val = self._mutual_info_xx(fi, fj)
                I_xx[(fi, fj)] = val
                pair_list.append(val)
        avg_I_xx = np.mean(pair_list) + self.eps if pair_list else self.eps

        for i, fid in enumerate(fids):
            self.D_cache[fid] = I_xy[fid] / avg_I_xy
            r_vals = []
            for j, fj in enumerate(fids):
                if i == j:
                    continue
                key = (fid, fj) if (fid, fj) in I_xx else (fj, fid)
                r_vals.append(I_xx.get(key, 0.0))
            self.R_cache[fid] = (np.mean(r_vals) / avg_I_xx) if r_vals else 0.0

    def _compute_weight(self, fid):
        d = self.D_cache.get(fid, 0.0)
        r = self.R_cache.get(fid, 0.0)
        a = self.alpha.get(fid, 1.0)
        b = self.beta.get(fid, 1.0)
        z = a * d - b * r
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def _optimize_coefficients(self):
        if len(self.recent_samples) < 10:
            return
        fids = list(self.feature_seen)
        if not fids:
            return

        alpha_arr = np.array([self.alpha.get(f, 1.0) for f in fids])
        beta_arr = np.array([self.beta.get(f, 1.0) for f in fids])
        gamma_val = float(self.gamma)

        D_arr = np.array([self.D_cache.get(f, 0.0) for f in fids])
        R_arr = np.array([self.R_cache.get(f, 0.0) for f in fids])
        log_prior = np.log(
            self.class_counts / (self.class_counts.sum() + self.eps) + self.eps)

        M = len(self.recent_samples)
        n_feat = len(fids)
        loglik = np.zeros((M, n_feat, self.num_classes))
        labels = np.zeros(M, dtype=int)
        fid2idx = {f: i for i, f in enumerate(fids)}

        for s_idx, (feat_dict, y) in enumerate(self.recent_samples):
            labels[s_idx] = y
            for fid, val in feat_dict.items():
                if fid not in fid2idx:
                    continue
                fi = fid2idx[fid]
                means = self.feature_means[fid]
                vars_ = self.feature_vars[fid] + self.eps
                ll = -0.5 * np.log(2 * np.pi * vars_) - 0.5 * (
                    (val - means) ** 2 / vars_)
                loglik[s_idx, fi, :] = ll

        def loss_and_grad(alpha, beta, gamma):
            z = alpha * D_arr - beta * R_arr
            w = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            weighted_ll = loglik * w[None, :, None]
            logits = log_prior[None, :] + gamma * weighted_ll.sum(axis=1)
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + self.eps)
            Y = np.zeros((M, self.num_classes))
            Y[np.arange(M), labels] = 1.0
            diff = probs - Y
            loss = 0.5 * np.sum(diff ** 2)
            dgamma = np.sum(diff * weighted_ll.sum(axis=1))
            w_grad = w * (1 - w)
            dalpha = np.zeros(n_feat)
            dbeta = np.zeros(n_feat)
            for i in range(n_feat):
                dL_dw_i = np.sum(diff * gamma * loglik[:, i, :])
                dalpha[i] = dL_dw_i * w_grad[i] * D_arr[i]
                dbeta[i] = dL_dw_i * w_grad[i] * (-R_arr[i])
            return loss, dalpha, dbeta, dgamma

        for _ in range(self.n_gd_iter):
            loss, dalpha, dbeta, dgamma = loss_and_grad(
                alpha_arr, beta_arr, gamma_val)
            alpha_arr -= self.lr * dalpha
            beta_arr -= self.lr * dbeta
            gamma_val -= self.lr * dgamma
            alpha_arr = np.clip(alpha_arr, 0.0, 10.0)
            beta_arr = np.clip(beta_arr, 0.0, 10.0)
            gamma_val = np.clip(gamma_val, 0.0, 10.0)

        self.loss_history.append(float(loss))
        for i, fid in enumerate(fids):
            self.alpha[fid] = float(alpha_arr[i])
            self.beta[fid] = float(beta_arr[i])
        self.gamma = float(gamma_val)

    def predict(self, X):
        if not self.is_fitted:
            return int(self.rng.randint(self.num_classes))

        if isinstance(X, (list, tuple)):
            X = np.array(X, dtype=float)
        feat_dict = {i: float(X[i]) for i in range(len(X))
                     if not np.isnan(X[i])}

        log_probs = np.log(
            self.class_counts / (self.class_counts.sum() + self.eps) + self.eps)

        for fid, val in feat_dict.items():
            if fid not in self.feature_seen:
                continue
            means = self.feature_means[fid]
            vars_ = self.feature_vars[fid] + self.eps
            log_lik = -0.5 * np.log(2 * np.pi * vars_) - 0.5 * (
                (val - means) ** 2 / vars_)
            w = self._compute_weight(fid)
            log_probs = log_probs + self.gamma * w * log_lik

        return int(np.argmax(log_probs))

    def reset(self):
        self.__init__(
            num_classes=self.num_classes,
            lr=self.lr,
            random_state=self.random_state,
            n_gd_iter=self.n_gd_iter,
            eps=self.eps,
            alpha_range=self.alpha_range,
            beta_range=self.beta_range,
            gamma_range=self.gamma_range,
        )