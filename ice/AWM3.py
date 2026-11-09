import numpy as np
from skmultiflow.trees import HoeffdingTree
from skmultiflow.drift_detection import ADWIN
from sklearn.feature_selection import mutual_info_classif


class ResidualLinearModel:
    def __init__(self, input_dim, n_classes, lr=0.002, l2_reg=1e-3, max_norm=2.0, rng=None):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.lr = lr
        self.l2_reg = l2_reg
        self.max_norm = max_norm
        self.rng = rng if rng is not None else np.random.RandomState()
        self.W = np.zeros((n_classes, input_dim), dtype=np.float64)
        self.b = np.zeros(n_classes, dtype=np.float64)

    def align_to(self, new_dim):
        w_dim = self.W.shape[1]
        if new_dim > w_dim:
            pad = np.zeros((self.n_classes, new_dim - w_dim), dtype=np.float64)
            self.W = np.hstack([self.W, pad])
            self.input_dim = new_dim

    def predict_logits(self, x_new):
        x_flat = np.asarray(x_new, dtype=np.float64).flatten()
        if len(x_flat) > self.W.shape[1]:
            self.align_to(len(x_flat))
        return self.W @ x_flat + self.b

    def partial_fit_ogd(self, x_new, p_base, y_true, damping_factor=1.0):
        x_flat = np.asarray(x_new, dtype=np.float64).flatten()
        self.align_to(len(x_flat))
        if len(x_flat) < self.W.shape[1]:
            x_flat = np.pad(
                x_flat,
                (0, self.W.shape[1] - len(x_flat)),
                mode='constant'
            )
        p_base = np.asarray(
            p_base,
            dtype=np.float64
        ).flatten()

        p_base = np.clip(
            p_base,
            1e-12,
            1.0
        )
        z_base = np.log(p_base)
        r = self.W @ x_flat + self.b

        z_combined = (
                z_base +
                damping_factor * r
        )
        z_shift = z_combined - np.max(z_combined)
        exp_z = np.exp(z_shift)

        p_combined = (
                exp_z /
                (np.sum(exp_z) + 1e-12)
        )
        y_onehot = np.zeros(
            self.n_classes,
            dtype=np.float64
        )
        y_onehot[int(y_true)] = 1.0
        ce_loss = -np.log(
            np.clip(
                p_combined[int(y_true)],
                1e-12,
                1.0
            )
        )
        reg_loss = (
                0.5 *
                self.l2_reg *
                np.sum(self.W ** 2)
        )

        loss = ce_loss + reg_loss
        error = p_combined - y_onehot

        grad_W = (
                damping_factor *
                np.outer(error, x_flat)
                +
                self.l2_reg * self.W
        )

        grad_b = (
                damping_factor *
                error
        )
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        self._project()

        return p_combined, loss

    def _project(self):
        norm = np.sqrt(
            np.sum(self.W ** 2) +
            np.sum(self.b ** 2)
        )

        if norm > self.max_norm:
            scale = self.max_norm / (norm + 1e-12)

            self.W *= scale
            self.b *= scale


class TreeExpert:
    def __init__(self, n_classes, use_residual=True, random_state=42,
                 tree_kwargs=None, residual_kwargs=None, damping_factor=1.0,
                 discrete_features=None, mi_threshold=0.05, mi_win_size=20, expert_id=None):
        self.n_classes = n_classes
        self.use_residual = use_residual
        self.damping_factor = 1.0
        self.discrete_features = discrete_features
        self.rng = np.random.RandomState(random_state)

        tree_kwargs = tree_kwargs or dict(
            split_confidence=0.5, grace_period=100, leaf_prediction="nb"
        )
        self._tree_kwargs = tree_kwargs
        self.tree = HoeffdingTree(**self._tree_kwargs)
        self.residual_model = None
        self._residual_kwargs = residual_kwargs or {}
        self.tree_dim = None
        self.current_feature_dim = 0
        self.expert_weights = np.array([0.5, 0.5])

        self.mi_feat = []
        self.mi_label = []
        self.mi_win_size = mi_win_size
        self.mi_threshold = mi_threshold
        self.mi_min_samples = 20
        self.mi_eval_budget = 0
        self.mi_eval_max = 500
        self.mi_last_dim = 0
        self.expert_id = expert_id
        self.residual_samples_seen = 0
        self.residual_loss_history = []
        self.cumulative_ogd_loss=0.0

    def partial_fit(self, x, y, classes=None):
        x = np.asarray(x, dtype=np.float64).flatten()
        cur_dim = len(x)
        if self.tree_dim is None:
            self.tree_dim=cur_dim
        if self.tree_dim is None or cur_dim >  self.current_feature_dim:
            if self.tree_dim is not None:
                self.mi_eval_budget =1
        self.current_feature_dim = cur_dim
        x_old = x[:self.tree_dim].reshape(1, -1)


        x_new = x[self.tree_dim:]
        cur_new_dim = len(x_new)

        if cur_new_dim > 0 and cur_new_dim > self.mi_last_dim:
            self.mi_eval_budget = self.mi_eval_max
            self.mi_last_dim = cur_new_dim
            self.mi_feat.clear()
            self.mi_label.clear()

        if self.use_residual and cur_new_dim > 0:
            if self.residual_model is None:
                self.residual_model = ResidualLinearModel(
                    input_dim=cur_new_dim,
                    n_classes=self.n_classes,
                    rng=self.rng,
                    **self._residual_kwargs,
                )
            _, p0, p1 = self.predict_proba(x)

            p_combined_before, residual_loss = (
                self.residual_model.partial_fit_ogd(
                    x_new=x_new,
                    p_base=p0,
                    y_true=y,
                    damping_factor=self.damping_factor
                )
            )
            self.residual_loss_history.append(residual_loss)
            self.cumulative_ogd_loss += residual_loss
            self.residual_samples_seen += 1

            self.mi_feat.append(np.asarray(x_new, dtype=np.float64).flatten())
            self.mi_label.append(int(y))
            if len(self.mi_feat) > self.mi_win_size:
                self.mi_feat = self.mi_feat[-self.mi_win_size:]
                self.mi_label = self.mi_label[-self.mi_win_size:]

            if self.mi_eval_budget > 0 and len(self.mi_feat) >= self.mi_min_samples:
                self._evaluate_new_features()
                self.mi_eval_budget = 0
        self.tree.partial_fit(x_old, y, classes=classes)
    def _evaluate_new_features(self):
        if len(self.mi_feat) < self.mi_min_samples:
            return False
        max_dim = max(len(f) for f in self.mi_feat)
        X_mi = np.array([
            np.pad(f, (0, max_dim - len(f)), mode='constant')
            for f in self.mi_feat
        ])
        y_mi = np.array(self.mi_label)
        if len(np.unique(y_mi)) < 2:
            return False

        disc_arg = self.discrete_features if isinstance(self.discrete_features, bool) else False

        try:
            mi_values = mutual_info_classif(X_mi, y_mi, discrete_features=disc_arg)
            mi_score = float(np.mean(mi_values))
            print(mi_score)

            if mi_score > self.mi_threshold:

                self._reset_expert(self.current_feature_dim)

        except Exception:
            pass
        self.mi_feat.clear()
        self.mi_label.clear()
        return mi_score

    def _reset_expert(self,tree_dim=None):
        self.tree = HoeffdingTree(**self._tree_kwargs)
        self.residual_model = None
        self.residual_samples_seen = 0
        self.expert_weights = np.array([0.5, 0.5])
        self.adwin = ADWIN(delta=0.01)
        self.tree_dim = tree_dim

    @staticmethod
    def _softmax(logits):
        exps = np.exp(logits - np.max(logits))
        return exps / (np.sum(exps) + 1e-12)

    def predict_proba(self, x):
        x = np.asarray(x, dtype=np.float64).flatten()
        if self.tree_dim is None:
            self.tree_dim = len(x)
        x_old = x[:self.tree_dim].reshape(1, -1)
        x_new = x[self.tree_dim:]
        prob_b = self.tree.predict_proba(x_old)[0]
        prob_b = np.asarray(prob_b, dtype=np.float64).flatten()
        if len(prob_b) < self.n_classes:
            prob_b = np.pad(prob_b, (0, self.n_classes - len(prob_b)), mode='constant')
        logit_b = np.log(np.clip(prob_b, 1e-12, 1.0 - 1e-12))
        p0 = self._softmax(logit_b)

        if (not self.use_residual) or (self.residual_model is None) or len(x_new) == 0:
            return p0, p0, p0
        logit_r = self.residual_model.predict_logits(x_new)
        p1 = self._softmax(logit_b +  self.damping_factor * logit_r)
        w = self.expert_weights / (np.sum(self.expert_weights) + 1e-12)
        prob_final = w[0] * p0 + w[1] * p1
        return prob_final, p0, p1

    def update_hedge(self, y_true, p0, p1, eta_hedge=0.005):

        y_onehot = np.zeros(self.n_classes)
        y_onehot[int(y_true)] = 1.0

        loss0 = np.mean((p0 - y_onehot) ** 2)
        loss1 = np.mean((p1 - y_onehot) ** 2)
        losses = np.array([loss0, loss1])
        self.expert_weights *= np.exp(-eta_hedge * losses)

        weight_sum = np.sum(self.expert_weights)

        if weight_sum > 0:
            self.expert_weights /= weight_sum
        else:
            self.expert_weights[:] = 0.5

        return loss0, loss1


class AWM_tree_mutil_Drift:
    def __init__(self, n_estimators=5, n_classes=2,
                 use_residual=True, random_state=None,
                 eta_hedge=0.005, damping_factor=0.1,mi_threshold=0.1,
                 discrete_features=None):
        self.n_estimators = n_estimators
        self.n_classes = n_classes
        self.eta_hedge = eta_hedge
        self.discrete_features = discrete_features
        self.rng = np.random.RandomState(random_state)
        self.adwin = ADWIN()

        self.experts = [
            TreeExpert(
                expert_id=i,
                n_classes=n_classes,
                use_residual=use_residual,
                random_state=self.rng.randint(0, 1_000_000),
                damping_factor=damping_factor,
                discrete_features=discrete_features,
                mi_threshold=mi_threshold
            )
            for i in range(n_estimators)
        ]

        self.initial_feature_dim = None
        self.global_feature_dim = None

        self.T = 0
        self.correct = 0
        self.cumulative_loss0 = 0.0
        self.cumulative_loss1 = 0.0
        self.cumulative_loss_awm = 0.0

        self.sample_history = []
        self.loss0_history = []
        self.loss1_history = []
        self.loss_awm_history = []
        self.delta_history = []
        self.acc_history = []
        self.record_interval = 500

    def partial_fit(self, X, y, classes=None):
        X = np.asarray(X, dtype=np.float64).flatten()
        current_dim = len(X)

        if self.initial_feature_dim is None:
            self.initial_feature_dim = current_dim
            self.global_feature_dim = current_dim


        k = 1
        k = min(k, self.n_estimators)
        idx_list = self.rng.choice(self.n_estimators, size=k, replace=False)

        if current_dim > self.global_feature_dim:

            self.global_feature_dim = current_dim


        for idx in idx_list:
            prob_final, p0, p1 = self.experts[idx].predict_proba(X)

            y_onehot = np.zeros(self.n_classes)
            y_onehot[int(y)] = 1.0

            if self.experts[idx].residual_model is not None:
                self.T += 1
                self.experts[idx].update_hedge(
                    y_true=y,
                    p0=p0,
                    p1=p1,
                    eta_hedge=self.eta_hedge
                )

            pred = int(np.argmax(prob_final))
            self.experts[idx].partial_fit(X, y, classes=classes)




    def predict(self, X):
        votes = []
        for e in self.experts:
            prob_final, p0, p1 = e.predict_proba(X)
            votes.append(int(np.argmax(prob_final)))
        final_pred = max(set(votes), key=votes.count)
        return [final_pred]