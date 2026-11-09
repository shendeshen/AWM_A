import numpy as np
import matplotlib.pyplot as plt
import time
import random

from model import AWM_tree_mutil
from IIF import IIFStreamClassifier
from OLSF import BaggingOLSF
from RVFL import LiteRVFL_Online
from GCMCC import GMMComponentClassifier
from rail import RAIL
from skmultiflow.meta import OzaBaggingAdwin

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


DATA_PATH   = "../LVL/data06/tree_g_DCC.csv"
WINDOW      = 500
N_REPEATS   = 15
SEEDS       = list(range(42, 42 + N_REPEATS))

CLASSES     = [0, 1, 2, 3, 4, 5, 6, 7]
ALG_NAMES   = ['awm', 'iif', 'olsf', 'obag', 'litervfl', 'gmmcomp', 'rail']

OLSF_CONFIG = {'variant': 'I', 'C': 1.0, 'lamda': 30.0, 'B': 0.5}
PREHEAT     = 500

LITER_VFL_CONFIG = dict(
    n_hidden=100,
    reg=1e-6,
    forgetting_rate=0.995,
    activation='sigmoid',
    n_classes=len(CLASSES),
)

GMM_CONFIG = dict(
    n_classes=len(CLASSES),
    C_f=1.0,
    corr_threshold=0.3,
)

RAIL_CONFIG = dict(
    num_classes=len(CLASSES),
    lr=0.01,
    n_gd_iter=10,
)


def read_csv(path):
    with open(path) as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            x = np.array([float(p) for p in parts[:-1]], dtype=np.float64)
            y = int(float(parts[-1]))
            yield x, y


def detect_max_dim(path):
    max_d = 0
    with open(path) as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            max_d = max(max_d, len(parts) - 1)
    return max_d


def build_feature_dims(d):
    if d <= 1:
        return [d]
    n_parts = min(3, d)
    base = d // n_parts
    dims = [base] * n_parts
    dims[-1] += d - sum(dims)
    return dims


def get_fixed_dimension_features(x, fixed_dim):
    d = len(x)
    if d < fixed_dim:
        x_pad = np.zeros(fixed_dim, dtype=np.float64)
        x_pad[:d] = x
        return x_pad.reshape(1, -1)
    return x[:fixed_dim].reshape(1, -1)


MAX_DIM      = detect_max_dim(DATA_PATH)
FEATURE_DIMS = build_feature_dims(MAX_DIM)
print(f"[INFO] Maximum feature dimension of the dataset MAX_DIM = {MAX_DIM}")
print(f"[INFO] IIF fixed feature partition FEATURE_DIMS = {FEATURE_DIMS}")


def run_single_experiment(seed):
    np.random.seed(seed)
    random.seed(seed)

    awm = AWM_tree_mutil(n_estimators=5, n_classes=len(CLASSES), random_state=seed)

    olsf = BaggingOLSF(variant=OLSF_CONFIG['variant'],
                       C=OLSF_CONFIG['C'],
                       lamda=OLSF_CONFIG['lamda'],
                       B=OLSF_CONFIG['B'],
                       random_state=seed)

    obag = OzaBaggingAdwin(n_estimators=2, random_state=seed)
    obag_dim = None
    obag_reset_count = 0

    litervfl = LiteRVFL_Online(random_state=seed, **LITER_VFL_CONFIG)
    lrvfl_dim = None
    lrvfl_reset_count = 0

    iif = IIFStreamClassifier(buffer_size=32, C=0.01, kernel='rbf',
                              gamma=0.1, B=50, random_state=seed)

    gmmcomp = GMMComponentClassifier(random_state=seed, **GMM_CONFIG)
    gmm_dim = None
    gmm_reset_count = 0

    # RAIL 初始化（不再重置）
    rail = RAIL(random_state=seed, **RAIL_CONFIG)
    rail_dim = None

    total       = 0
    err         = {alg: [] for alg in ALG_NAMES}
    acc_windows = {alg: [] for alg in ALG_NAMES}
    time_cost   = {alg: 0.0 for alg in ALG_NAMES}

    for x, y in read_csv(DATA_PATH):
        total += 1
        current_dim = x.shape[0]

        if obag_dim is None:
            obag_dim = current_dim
        elif current_dim > obag_dim:
            print(f"[OBag] Feature dimension increased {obag_dim} -> {current_dim}, resetting OzaBaggingAdwin")
            obag = OzaBaggingAdwin(n_estimators=2, random_state=seed)
            obag_dim = current_dim
            obag_reset_count += 1

        if lrvfl_dim is None:
            lrvfl_dim = current_dim
        elif current_dim > lrvfl_dim:
            print(f"[LiteRVFL] Feature dimension increased {lrvfl_dim} -> {current_dim}, resetting LiteRVFL_Online")
            litervfl = LiteRVFL_Online(random_state=seed, **LITER_VFL_CONFIG)
            lrvfl_dim = current_dim
            lrvfl_reset_count += 1

        if gmm_dim is None:
            gmm_dim = current_dim
            gmmcomp.reset(current_dim)
        elif current_dim != gmm_dim:
            print(f"[GMMComponent] Feature dimension changed {gmm_dim} -> {current_dim}, resetting")
            gmmcomp.reset(current_dim)
            gmm_dim = current_dim
            gmm_reset_count += 1

        # --- 注意这里 ---
        # 仅更新记录的维度，不再重建 RAIL 模型
        if rail_dim is None:
            rail_dim = current_dim
        elif current_dim != rail_dim:
            # print(f"[RAIL] Feature dimension changed {rail_dim} -> {current_dim}, RAIL dynamically handles this without reset.")
            rail_dim = current_dim

        X       = x.reshape(1, -1)
        y_arr   = np.array([y])
        X_fixed = get_fixed_dimension_features(x, 2)

        x_iif = np.zeros(MAX_DIM, dtype=np.float64)
        x_iif[:len(x)] = x

        if total <= PREHEAT:
            start = time.time(); awm.partial_fit(X, y_arr, classes=CLASSES); time_cost['awm'] += time.time() - start
            start = time.time(); olsf.partial_fit(x, y); time_cost['olsf'] += time.time() - start
            start = time.time(); obag.partial_fit(X, y_arr, classes=CLASSES); time_cost['obag'] += time.time() - start
            start = time.time(); iif.partial_fit(x_iif, y, feature_dims=FEATURE_DIMS); time_cost['iif'] += time.time() - start
            start = time.time(); litervfl.partial_fit(X, y_arr); time_cost['litervfl'] += time.time() - start
            start = time.time(); gmmcomp.partial_fit(X, y_arr, classes=CLASSES); time_cost['gmmcomp'] += time.time() - start
            start = time.time(); rail.partial_fit(x, y); time_cost['rail'] += time.time() - start
            continue

        start = time.time()
        p_awm = awm.predict(X)[0]
        awm.partial_fit(X, y_arr, classes=CLASSES)
        time_cost['awm'] += time.time() - start
        err['awm'].append(0 if p_awm == y else 1)

        start = time.time()
        p_olsf = olsf.predict(x)
        olsf.partial_fit(x, y)
        time_cost['olsf'] += time.time() - start
        err['olsf'].append(0 if p_olsf == y else 1)

        start = time.time()
        p_obag = obag.predict(X)[0]
        obag.partial_fit(X, y_arr, classes=CLASSES)
        time_cost['obag'] += time.time() - start
        err['obag'].append(0 if p_obag == y else 1)

        start = time.time()
        p_iif_raw = iif.predict(x_iif, feature_dims=FEATURE_DIMS)
        p_iif = 1 if p_iif_raw == 1 else 0
        iif.partial_fit(x_iif, y, feature_dims=FEATURE_DIMS)
        time_cost['iif'] += time.time() - start
        err['iif'].append(0 if p_iif == y else 1)

        start = time.time()
        p_lrvfl = litervfl.predict(X)[0]
        litervfl.partial_fit(X, y_arr)
        time_cost['litervfl'] += time.time() - start
        err['litervfl'].append(0 if p_lrvfl == y else 1)

        start = time.time()
        p_gmm = gmmcomp.predict(X)[0]
        gmmcomp.partial_fit(X, y_arr, classes=CLASSES)
        time_cost['gmmcomp'] += time.time() - start
        err['gmmcomp'].append(0 if p_gmm == y else 1)

        start = time.time()
        p_rail = rail.predict(x)
        rail.partial_fit(x, y)
        time_cost['rail'] += time.time() - start
        err['rail'].append(0 if p_rail == y else 1)

        if len(err['awm']) >= WINDOW:
            for alg in ALG_NAMES:
                acc_windows[alg].append(1.0 - float(np.mean(err[alg])))
                err[alg].clear()

    print(f"[OBag]     Total resets in this run: {obag_reset_count}")
    print(f"[LiteRVFL] Total resets in this run: {lrvfl_reset_count}")
    print(f"[GMMComp]  Total resets in this run: {gmm_reset_count}")


    final_acc = {alg: (float(np.mean(acc_windows[alg])) if acc_windows[alg] else 0.0)
                 for alg in ALG_NAMES}
    return final_acc, time_cost, acc_windows


def plot_acc_curves(all_curves, save_path='acc_curves_argg.png'):
    plt.figure(figsize=(12, 6.5))

    solid_algs = ['awm', 'iif', 'olsf', 'rail']
    dashed_algs = ['obag', 'litervfl', 'gmmcomp']

    style_map = {
        'awm':      dict(color='#1f77b4', linestyle='-',  linewidth=2.0),
        'iif':      dict(color='#d62728', linestyle='-',  linewidth=2.0),
        'olsf':     dict(color='#2ca02c', linestyle='-',  linewidth=2.0),
        'rail':     dict(color='#9467bd', linestyle='-',  linewidth=2.0),
        'obag':     dict(color='#ff7f0e', linestyle='--', linewidth=1.8),
        'litervfl': dict(color='#8c564b', linestyle='--', linewidth=1.8),
        'gmmcomp':  dict(color='#e377c2', linestyle='--', linewidth=1.8),
    }

    for alg in ALG_NAMES:
        curves = all_curves[alg]
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        if min_len == 0:
            continue
        arr = np.array([np.asarray(c[:min_len], dtype=float) for c in curves])
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        x_axis = np.arange(1, min_len + 1) * WINDOW

        style = style_map.get(alg, dict(color='gray', linestyle='-', linewidth=1.5))

        dash_pattern = None
        if alg == 'obag':
            dash_pattern = (0, (5, 2))
        elif alg == 'litervfl':
            dash_pattern = (0, (3, 1, 1, 1))
        elif alg == 'gmmcomp':
            dash_pattern = (0, (1, 1))

        plt.plot(
            x_axis, mean,
            label=alg.upper(),
            color=style['color'],
            linestyle=dash_pattern if dash_pattern else style['linestyle'],
            linewidth=style['linewidth'],
        )
        plt.fill_between(x_axis, mean - std, mean + std, color=style['color'], alpha=0.12)

    plt.xlabel('Stream samples', fontsize=12)
    plt.ylabel('Window accuracy', fontsize=12)
    plt.title('Accuracy curves of online learning algorithms', fontsize=13)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=style_map['awm']['color'],      lw=2.0, linestyle='-',               label='AWM'),
        Line2D([0], [0], color=style_map['iif']['color'],      lw=2.0, linestyle='-',               label='IIF'),
        Line2D([0], [0], color=style_map['olsf']['color'],     lw=2.0, linestyle='-',               label='OLSF'),
        Line2D([0], [0], color=style_map['rail']['color'],     lw=2.0, linestyle='-',               label='RAIL'),
        Line2D([0], [0], color=style_map['obag']['color'],     lw=1.8, linestyle=(0, (5, 2)),       label='OBAG'),
        Line2D([0], [0], color=style_map['litervfl']['color'], lw=1.8, linestyle=(0, (3, 1, 1, 1)), label='LiteRVFL'),
        Line2D([0], [0], color=style_map['gmmcomp']['color'],  lw=1.8, linestyle=(0, (1, 1)),       label='GMMCOMP'),
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nAccuracy curves saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    all_results = {alg: [] for alg in ALG_NAMES}
    all_times   = {alg: [] for alg in ALG_NAMES}
    all_curves  = {alg: [] for alg in ALG_NAMES}

    print(f"\nStarting repeated experiments ({N_REPEATS} runs in total)...")
    for idx, seed in enumerate(SEEDS):
        print(f"\n--- Run {idx + 1}/{N_REPEATS}, random seed = {seed} ---")
        acc, times, curves = run_single_experiment(seed)

        for alg in ALG_NAMES:
            all_results[alg].append(acc[alg])
            all_times[alg].append(times[alg])
            all_curves[alg].append(curves[alg])

        print("Current accuracy: " + ", ".join(f"{a.upper()}={acc[a]:.4f}" for a in ALG_NAMES))

    print("\n" + "=" * 80)
    print(f"Final results ({N_REPEATS} runs, mean ± std)")
    print("=" * 80)
    for alg in ALG_NAMES:
        print(f"{alg.upper():9s} : {np.mean(all_results[alg]):.4f} ± {np.std(all_results[alg]):.4f}")

    print("\n" + "=" * 80)
    print("Running time statistics (total seconds, mean ± std)")
    print("=" * 80)
    for alg in ALG_NAMES:
        print(f"{alg.upper():9s} : {np.mean(all_times[alg]):.2f} ± {np.std(all_times[alg]):.2f} s")

    plot_acc_curves(all_curves, save_path='acc_curves.png')

    print("\nExperiment completed.")
