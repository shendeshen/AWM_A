import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from collections import deque


class kDNFilter:

    def __init__(self, k=5, threshold=0.8):
        self.k = k
        self.threshold = threshold

    def compute_kdn(self, X, y):
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        _, indices = nbrs.kneighbors(X)

        scores = np.zeros(len(X))
        for i in range(len(X)):
            neighbors = indices[i][1:]
            scores[i] = np.sum(y[neighbors] != y[i]) / self.k

        return scores

    def filter(self, X, y):
        scores = self.compute_kdn(X, y)
        mask = scores < self.threshold
        return mask, scores


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, dist, label):
        loss_pos = label * dist.pow(2)
        loss_neg = (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        return (loss_pos + loss_neg).mean() / 2


class EmbeddingNetwork(nn.Module):

    def __init__(self, input_dim, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class PCADriftDetector:

    def __init__(self, n_components=None, threshold=0.5):
        self.n_components = n_components
        self.threshold = threshold
        self.pca = None
        self.ref_stats = None

    def fit(self, X_ref):
        n_pc = self.n_components or min(X_ref.shape[1], 10)
        self.pca = PCA(n_components=n_pc)
        Z = self.pca.fit_transform(X_ref)

        self.ref_stats = {
            'mean': Z.mean(axis=0),
            'var': Z.var(axis=0) + 1e-8
        }

    def detect(self, X_new):
        if self.pca is None:
            raise ValueError("Please call fit() first to fit reference data")

        Z_new = self.pca.transform(X_new)
        mu_new = Z_new.mean(axis=0)
        var_new = Z_new.var(axis=0) + 1e-9

        mu_ref = self.ref_stats['mean']
        var_ref = self.ref_stats['var']

        kl = 0.5 * (
                np.log(var_ref / var_new) +
                (var_new + (mu_new - mu_ref) ** 2) / var_ref -1
        ).sum()

        return kl > self.threshold, kl


class GMMComponent:

    def __init__(self, dim, init_mean=None, init_cov=None, weight=1.0,
                 class_label=0, C_f=1.0):
        self.dim = dim
        self.weight = weight
        self.mean = init_mean if init_mean is not None else np.zeros(dim)
        self.cov = init_cov if init_cov is not None else C_f * np.eye(dim)
        self.sp = weight
        self.class_label = class_label
        self.C_f = C_f

    def pdf(self, x):
        diff = x - self.mean
        try:
            inv_cov = np.linalg.inv(self.cov)
            det_cov = np.linalg.det(self.cov)
            if det_cov <= 0:
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(self.cov)
            det_cov = max(abs(np.linalg.det(self.cov)), 1e-10)

        exponent = -0.4 * diff @ inv_cov @ diff
        norm = 1.0 / ((2 * np.pi) ** (self.dim / 2) * np.sqrt(det_cov))
        return norm * np.exp(np.clip(exponent, -200, 0))

    def posterior(self, x, all_components):
        numer = self.pdf(x) * self.weight
        denom = sum(c.pdf(x) * c.weight for c in all_components)
        return numer / (denom + 1e-10)

    def update(self, x, posterior):
        self.sp += posterior
        old_mean = self.mean.copy()
        self.mean = old_mean + (posterior / self.sp) * (x - old_mean)
        diff = (x - self.mean).reshape(-1, 1)
        self.cov = self.cov + (posterior / self.sp) * (diff @ diff.T - self.cov)

        self.cov = 0.5 * (self.cov + self.cov.T)
        self.cov += 1e-9 * np.eye(self.dim)


class GMMComponentClassifier:

    def __init__(self, n_classes, C_f=1.0, corr_threshold=0.2,
                 random_state=None):
        self.n_classes = n_classes
        self.C_f = C_f
        self.corr_threshold = corr_threshold
        self.components = {}
        self.dim = None
        self.rng = np.random.RandomState(random_state)

    def reset(self, dim):
        self.components = {}
        self.dim = dim

    def partial_fit(self, X, y, classes=None):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        if self.dim is None or X.shape[1] != self.dim:
            self.reset(X.shape[1])

        for xi, yi in zip(X, y):
            yi = int(yi)
            if yi not in self.components:
                init_mean = xi + self.rng.normal(0, 0.1, size=self.dim)
                diag = self.C_f * (1.0 + 0.5 * self.rng.randn(self.dim))
                init_cov = np.diag(np.abs(diag) + 1e-6)

                self.components[yi] = GMMComponent(
                    dim=self.dim,
                    init_mean=init_mean,
                    init_cov=init_cov,
                    weight=1.0,
                    class_label=yi,
                    C_f=self.C_f
                )
            else:
                comp = self.components[yi]
                posterior = 1.0 * (1.0 + 0.05 * self.rng.randn())
                posterior = np.clip(posterior, 0.01, 1.0)
                comp.update(xi, posterior)

        total_w = sum(c.weight for c in self.components.values())
        if total_w > 0:
            for c in self.components.values():
                c.weight /= total_w

    def predict(self, X):
        X = np.atleast_2d(X)
        if self.dim is None or X.shape[1] != self.dim or not self.components:
            return np.zeros(len(X), dtype=int)

        preds = []
        for xi in X:
            best_cls = 0
            best_score = -np.inf
            for cls, comp in self.components.items():
                score = comp.pdf(xi) * comp.weight
                if score > best_score:
                    best_score = score
                    best_cls = cls
            preds.append(best_cls)
        return np.array(preds)


class GMMPool:

    def __init__(self, dim, corr_threshold=0.3, C_f=1.0):
        self.dim = dim
        self.corr_threshold = corr_threshold
        self.C_f = C_f
        self.components = []
        self.n_classes = 0

    def fit_initial(self, X, y):
        self.n_classes = len(np.unique(y))
        for cls in np.unique(y):
            X_cls = X[y == cls]
            if len(X_cls) > 0:
                n_comp = min(3, len(X_cls))
                gmm = GaussianMixture(n_components=n_comp,
                                      covariance_type='full',
                                      random_state=42)
                gmm.fit(X_cls)

                for i in range(n_comp):
                    comp = GMMComponent(
                        dim=self.dim,
                        init_mean=gmm.means_[i],
                        init_cov=gmm.covariances_[i],
                        weight=gmm.weights_[i]
                    )
                    comp.class_label = cls
                    self.components.append(comp)

    def predict_proba(self, X):
        probas = np.zeros((len(X), len(self.components)))
        for j, comp in enumerate(self.components):
            probas[:, j] = np.array([comp.pdf(x) * comp.weight for x in X])
        probas = probas / (probas.sum(axis=1, keepdims=True) + 1e-10)
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        class_probs = np.zeros((len(X), self.n_classes))
        for j, comp in enumerate(self.components):
            class_probs[:, comp.class_label] += probas[:, j]
        return np.argmax(class_probs, axis=1)

    def update_virtual_drift(self, x, y_true=None):
        posteriors = []
        for comp in self.components:
            p = comp.posterior(x, self.components)
            posteriors.append(p)

        max_posterior = max(posteriors) if posteriors else 0

        if max_posterior > self.corr_threshold:
            best_idx = np.argmax(posteriors)
            self.components[best_idx].update(x, posteriors[best_idx])
        elif max_posterior == 0:
            new_comp = GMMComponent(
                dim=self.dim,
                init_mean=x,
                init_cov=self.C_f * np.eye(self.dim),
                weight=1.0
            )
            if y_true is not None:
                new_comp.class_label = y_true
            self.components.append(new_comp)

    def handle_real_drift(self, X_new, y_new):
        self.components = []
        self.fit_initial(X_new, y_new)


class GMMCL:

    def __init__(self,
                 window_size=200,
                 kdn_k=5,
                 kdn_threshold=0.8,
                 contrastive_margin=1.0,
                 pca_threshold=0.5,
                 corr_threshold=0.3,
                 C_f=1.0):

        self.window_size = window_size
        self.kdn_filter = kDNFilter(k=kdn_k, threshold=kdn_threshold)
        self.contrastive_loss = ContrastiveLoss(margin=contrastive_margin)
        self.pca_detector = PCADriftDetector(threshold=pca_threshold)
        self.corr_threshold = corr_threshold
        self.C_f = C_f

        self.gmm_pool = None
        self.embed_net = None
        self.optimizer = None
        self.buffer = deque(maxlen=window_size)
        self.is_initialized = False

    def _init_embedding_network(self, input_dim):
        self.embed_net = EmbeddingNetwork(input_dim, embed_dim=32)
        self.optimizer = torch.optim.Adam(self.embed_net.parameters(), lr=1e-3)

    def _train_contrastive(self, X_clean, y_clean):
        if self.embed_net is None:
            return

        X_tensor = torch.FloatTensor(X_clean)
        y_tensor = torch.LongTensor(y_clean)

        embeddings = self.embed_net(X_tensor)
        dist_matrix = torch.cdist(embeddings, embeddings)

        pos_mask = (y_tensor.unsqueeze(0) == y_tensor.unsqueeze(1))
        pos_mask.fill_diagonal_(False)

        if pos_mask.sum() > 0:
            pos_dist = dist_matrix[pos_mask].mean()
        else:
            pos_dist = torch.tensor(0.0)

        neg_mask = ~pos_mask
        neg_mask.fill_diagonal_(False)

        if neg_mask.sum() > 0:
            neg_dist = dist_matrix[neg_mask].mean()
        else:
            neg_dist = torch.tensor(0.0)

        loss = self.contrastive_loss(
            torch.stack([pos_dist, neg_dist]),
            torch.tensor([1.0, 0.0])
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _process_window(self, X_window, y_window):
        clean_mask, kdn_scores = self.kdn_filter.filter(X_window, y_window)
        X_clean = X_window[clean_mask]
        y_clean = y_window[clean_mask]

        if len(X_clean) == 0:
            return None

        if self.embed_net is None:
            self._init_embedding_network(X_clean.shape[1])
        self._train_contrastive(X_clean, y_clean)

        if not self.is_initialized:
            self.pca_detector.fit(X_clean)
            self.gmm_pool = GMMPool(
                dim=X_clean.shape[1],
                corr_threshold=self.corr_threshold,
                C_f=self.C_f
            )
            self.gmm_pool.fit_initial(X_clean, y_clean)
            self.is_initialized = True
            return 0.0

        is_drift, kl = self.pca_detector.detect(X_clean)

        if is_drift:
            self.gmm_pool.handle_real_drift(X_clean, y_clean)
            self.pca_detector.fit(X_clean)
        else:
            for x, y in zip(X_clean, y_clean):
                self.gmm_pool.update_virtual_drift(x, y)

        return kl

    def fit_predict_stream(self, X_stream, y_stream, batch_size=None):
        batch_size = batch_size or self.window_size
        accuracies = []

        for i in range(0, len(X_stream), batch_size):
            X_batch = X_stream[i:i + batch_size]
            y_batch = y_stream[i:i + batch_size]

            if len(X_batch) < 2:
                continue

            kl = self._process_window(X_batch, y_batch)

            if self.is_initialized and self.gmm_pool is not None:
                y_pred = self.gmm_pool.predict(X_batch)
                acc = np.mean(y_pred == y_batch)
                accuracies.append(acc)

                print(f"[Batch {i // batch_size}] "
                      f"Accuracy: {acc:.4f}, "
                      f"KL: {kl:.4f if kl else 0:.4f}, "
                      f"Components: {len(self.gmm_pool.components)}")

        return accuracies
