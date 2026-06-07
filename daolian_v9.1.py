import math, csv, json, os, sys, argparse
import numpy as np
from scipy.optimize import differential_evolution
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

ATOM_PARAMS = {
    'H': {'nf': 1, 'chi': 2.20, 'r_cov': 0.31},
    'C': {'nf': 3, 'chi': 2.55, 'r_cov': 0.76},
    'N': {'nf': 4, 'chi': 3.04, 'r_cov': 0.71},
    'O': {'nf': 5, 'chi': 3.44, 'r_cov': 0.66},
    'F': {'nf': 6, 'chi': 3.98, 'r_cov': 0.57},
}

BUILTIN_MOLECULES = [
    ('H2',   ['H','H'],                [(0,1,1)], 4.748),
    ('CH4',  ['C','H','H','H','H'],    [(0,1,1),(0,2,1),(0,3,1),(0,4,1)], 18.19),
    ('C2H6', ['C','C','H','H','H','H','H','H'],
     [(0,1,1),(0,2,1),(0,3,1),(0,4,1),(1,5,1),(1,6,1),(1,7,1)], 30.13),
    ('C2H4', ['C','C','H','H','H','H'],
     [(0,1,2),(0,2,1),(0,3,1),(1,4,1),(1,5,1)], 22.44),
    ('C2H2', ['C','C','H','H'],
     [(0,1,3),(0,2,1),(1,3,1)], 13.20),
    ('C3H8', ['C','C','C','H','H','H','H','H','H','H','H'],
     [(0,1,1),(1,2,1),(0,3,1),(0,4,1),(0,5,1),(1,6,1),(1,7,1),(2,8,1),(2,9,1),(2,10,1)], 36.93),
    ('C6H6', ['C','C','C','C','C','C','H','H','H','H','H','H'],
     [(0,1,1.5),(1,2,1.5),(2,3,1.5),(3,4,1.5),(4,5,1.5),(5,0,1.5),
      (0,6),(1,7),(2,8),(3,9),(4,10),(5,11)], 57.60),
    ('NH3',  ['N','H','H','H'],        [(0,1,1),(0,2,1),(0,3,1)], 12.86),
    ('H2O',  ['O','H','H'],            [(0,1,1),(0,2,1)], 10.09),
    ('HF',   ['H','F'],                [(0,1,1)], 5.87),
    ('F2',   ['F','F'],                [(0,1,1)], 1.66),
    ('H2O2', ['O','O','H','H'],        [(0,1,1),(0,2,1),(1,3,1)], 10.89),
    ('N2H4', ['N','N','H','H','H','H'], [(0,1,1),(0,2,1),(0,3,1),(1,4,1),(1,5,1)], 17.47),
    ('F2O',  ['O','F','F'],            [(0,1,1),(0,2,1)], 5.85),
    ('CO',   ['C','O'],                [(0,1,3)], 11.18),
    ('N2',   ['N','N'],                [(0,1,3)], 9.91),
    ('O2',   ['O','O'],                [(0,1,2)], 5.16),
    ('NO',   ['N','O'],                [(0,1,2)], 6.51),
    ('CO2',  ['C','O','O'],            [(0,1,2),(0,2,2)], 16.84),
    ('H2CO', ['C','O','H','H'],        [(0,1,2),(0,2,1),(0,3,1)], 16.20),
    ('HCN',  ['C','N','H'],            [(0,1,3),(0,2,1)], 13.61),
    ('CH3OH',['C','O','H','H','H','H'], [(0,1,1),(0,2,1),(0,3,1),(0,4,1),(1,5,1)], 24.28),
    ('CH3CN',['C','C','N','H','H','H'], [(0,1,1),(1,2,3),(0,3,1),(0,4,1),(0,5,1)], 28.15),
    ('CH3NH2',['C','N','H','H','H','H','H'], [(0,1,1),(0,2,1),(0,3,1),(0,4,1),(1,5,1),(1,6,1)], 23.30),
    ('CH3CHO',['C','C','O','H','H','H','H'], [(0,1,1),(1,2,2),(0,3,1),(0,4,1),(0,5,1),(1,6,1)], 26.01),
    ('C2H5OH',['C','C','O','H','H','H','H','H','H'],
     [(0,1,1),(1,2,1),(2,3,1),(0,4,1),(0,5,1),(0,6,1),(1,7,1),(1,8,1)], 28.95),
    ('HCOOH',['C','O','O','H','H'],    [(0,1,2),(0,2,1),(2,3,1),(0,4,1)], 19.84),
    ('CH3F', ['C','F','H','H','H'],    [(0,1,1),(0,2,1),(0,3,1),(0,4,1)], 22.60),
    ('CH3OCH3',['C','O','C','H','H','H','H','H','H'],
     [(0,1,1),(1,2,1),(0,3,1),(0,4,1),(0,5,1),(2,6,1),(2,7,1),(2,8,1)], 26.10),
    ('O3',   ['O','O','O'],            [(0,1,1.5),(1,2,1.5)], 6.25),
    ('甘氨酸',['C','C','N','O','O','H','H','H','H','H'],
     [(0,1,1),(1,2,1),(1,3,2),(1,4,1),(2,5,1),(2,6,1),(0,7,1),(0,8,1),(4,9,1)], 28.95),
]

DEFAULT_PHYS_PARAMS = np.array([
    3.817,
    0.028,
    0.010,
    0.078,
    0.020,
    0.150,
    0.862,
    0.850,
    0.880,
    0.195,
    0.102,
    0.500,
    0.000,
    -0.433,
    1.200,
    0.100,
    0.300,
    0.100,
    -0.500,
    0.100,
    0.200,
    0.500,
    0.300,
    0.200,
    0.000,
    0.000,
    0.400,
    0.500
])

PHYS_BOUNDS = [
    (3.0, 6.0),
    (0.028, 0.028),
    (0.01, 0.08),
    (0.01, 0.08),
    (0.02, 0.02),
    (0.15, 0.15),
    (0.6, 0.95),
    (0.5, 0.85),
    (0.6, 0.95),
    (0.0, 0.30),
    (0.1, 1.5),
    (0.5, 0.5),
    (0.0, 1.5),
    (-0.5, 0.5),
    (0.0, 3.0),
    (0.0, 0.5),
    (0.0, 0.8),
    (0.0, 1.0),
    (-2.0, 0.0),
    (0.0, 0.5),
    (0.0, 0.8),
    (0.2, 1.0),
    (0.0, 0.8),
    (0.0, 0.6),
    (-1.0, 1.0),
    (-0.5, 0.5),
    (0.0, 1.0),
    (0.0, 0.8)
]


N_FEATURES = 20
HIDDEN = 8
NN_PARAM_COUNT = (HIDDEN * N_FEATURES + HIDDEN) + (1 * HIDDEN + 1)


def extract_features(atoms, bonds):
    n = len(atoms)
    n_heavy = sum(1 for a in atoms if a != 'H')
    n_h = n - n_heavy
    nf_vals = [ATOM_PARAMS[a]['nf'] for a in atoms]
    chi_vals = [ATOM_PARAMS[a]['chi'] for a in atoms]
    r_vals = [ATOM_PARAMS[a]['r_cov'] for a in atoms]
    avg_nf = np.mean(nf_vals) if nf_vals else 0
    std_nf = np.std(nf_vals) if nf_vals else 0
    avg_chi = np.mean(chi_vals) if chi_vals else 0
    std_chi = np.std(chi_vals) if chi_vals else 0
    avg_r = np.mean(r_vals) if r_vals else 0
    total_bonds = len(bonds)
    if total_bonds == 0:
        avg_order = 0
        frac_multiple = 0
    else:
        orders = [bond[2] if len(bond)==3 else 1 for bond in bonds]
        avg_order = np.mean(orders)
        frac_multiple = sum(1 for o in orders if o > 1) / total_bonds
    chi_list = list(chi_vals)
    max_chi_diff = max(chi_list) - min(chi_list) if chi_list else 0
    h_donors = sum(1 for bond in bonds if (atoms[bond[0]]=='H' and atoms[bond[1]] in ('N','O','F')) or
                   (atoms[bond[1]]=='H' and atoms[bond[0]] in ('N','O','F')))
    lone_pairs = sum((ATOM_PARAMS[a]['nf']-2)//2 for a in atoms if a in ('N','O','F'))
    bond_counts = Counter()
    for bond in bonds:
        i, j = bond[0], bond[1]
        a1, a2 = atoms[i], atoms[j]
        if a1 > a2: a1, a2 = a2, a1
        bond_counts[f"{a1}-{a2}"] += 1
    target_bonds = ['C-H','C-C','C-O','C-N','O-H','N-H']
    bond_features = [bond_counts.get(b, 0) for b in target_bonds]
    return np.array([n, n_heavy, n_h, avg_nf, std_nf, avg_chi, std_chi, avg_r,
                     total_bonds, avg_order, frac_multiple, max_chi_diff,
                     h_donors, lone_pairs] + bond_features)


def daolian_energy(atoms, bonds, phys_params):
    (J0, alpha_shell, lambda_polar, lambda_lone, gamma_shell, lambda_bond,
     base_sat_double, base_sat_triple, base_sat_arom,
     func_coeff, mr_strength, polar_order_decay, hbond_strength, disp_coeff,
     arom_strength, ether_weaken, fluoro_polar_boost, ionic_enhance,
     linear_resonance, triple_decay, lone_shield_OH, ionic_nonlinear,
     carbonyl_polar_boost, amine_lone_shift, ring_stability, finite_size_corr,
     hetero_arom_decay, nitro_polar_boost) = phys_params
    n = len(atoms)
    m = len(bonds)
    D_e = 0.0
    arom_bonds = 0
    hetero_count = 0
    ether_pairs = set()
    for i in range(n):
        if atoms[i] == 'O':
            c_neighbors = []
            for a, b, _ in bonds:
                if a == i and atoms[b] == 'C':
                    c_neighbors.append(b)
                elif b == i and atoms[a] == 'C':
                    c_neighbors.append(a)
            if len(c_neighbors) == 2:
                ether_pairs.add((i, c_neighbors[0]))
                ether_pairs.add((i, c_neighbors[1]))
    for bond in bonds:
        if len(bond) == 2:
            i, j = bond; order = 1
        else:
            i, j, order = bond
        if order == 1.5:
            arom_bonds += 1
            if atoms[i] in ('N', 'O'): hetero_count += 1
            if atoms[j] in ('N', 'O'): hetero_count += 1
    heavy_atoms = [i for i, a in enumerate(atoms) if a != 'H']
    if len(heavy_atoms) == 3:
        mid = heavy_atoms[1]
        if atoms[mid] == 'C' and len([b for b in bonds if (b[0] == mid or b[1] == mid)]) == 2:
            end1, end2 = heavy_atoms[0], heavy_atoms[2]
            if atoms[end1] == atoms[end2] == 'O':
                orders = [b[2] if len(b) == 3 else 1 for b in bonds if
                          (b[0] == mid and b[1] in (end1, end2)) or (b[1] == mid and b[0] in (end1, end2))]
                if all(o == 2 for o in orders):
                    D_e += linear_resonance
    for bond in bonds:
        if len(bond) == 2:
            i, j = bond; order = 1
        else:
            i, j, order = bond
        a1, a2 = atoms[i], atoms[j]
        nf1, nf2 = ATOM_PARAMS[a1]['nf'], ATOM_PARAMS[a2]['nf']
        chi1, chi2 = ATOM_PARAMS[a1]['chi'], ATOM_PARAMS[a2]['chi']
        r1, r2 = ATOM_PARAMS[a1]['r_cov'], ATOM_PARAMS[a2]['r_cov']
        delta_chi = abs(chi1 - chi2)
        avg_nf = (nf1 + nf2) / 2.0
        shell_sensitivity = 1.0 + gamma_shell * (avg_nf - 1.0)
        effective_lambda = lambda_polar * (order ** (-polar_order_decay))
        is_NO2 = (order == 2) and ((a1 == 'N' and a2 == 'O') or (a1 == 'O' and a2 == 'N'))
        if is_NO2:
            effective_lambda = lambda_polar * (1.0 - nitro_polar_boost) * (order ** (-polar_order_decay))
        else:
            is_CO_double = (order == 2) and ((a1 == 'C' and a2 == 'O') or (a1 == 'O' and a2 == 'C'))
            if is_CO_double:
                effective_lambda = lambda_polar * (1.0 - carbonyl_polar_boost) * (order ** (-polar_order_decay))
            else:
                is_CF = (a1 == 'C' and a2 == 'F') or (a1 == 'F' and a2 == 'C')
                if is_CF:
                    effective_lambda = lambda_polar * (1.0 - fluoro_polar_boost) * (order ** (-polar_order_decay))
                else:
                    effective_lambda = lambda_polar * (order ** (-polar_order_decay))
        polar_factor = math.exp(-effective_lambda * delta_chi * shell_sensitivity)
        ionic_factor = 1.0
        if a1 != a2:
            ionic_factor = 1.0 + ionic_enhance * (delta_chi ** ionic_nonlinear)
        bond_length = r1 + r2
        ref_length = 1.07
        length_factor = math.exp(-lambda_bond * (bond_length - ref_length))
        if order == 1:
            sat = 1.0
        elif order == 2:
            sat = base_sat_double + alpha_shell * (avg_nf - 2.0)
        elif order == 3:
            sat = (base_sat_triple - triple_decay) + alpha_shell * (avg_nf - 2.0)
        elif order == 1.5:
            sat = base_sat_arom + alpha_shell * (avg_nf - 2.0)
        else:
            sat = 1.0
        lp1 = 0.0 if a1 not in ('N', 'O', 'F') else (ATOM_PARAMS[a1]['nf'] - 3) * 0.1
        lp2 = 0.0 if a2 not in ('N', 'O', 'F') else (ATOM_PARAMS[a2]['nf'] - 3) * 0.1
        is_OH = (a1 == 'O' and a2 == 'H') or (a1 == 'H' and a2 == 'O')
        if is_OH:
            if a1 == 'O':
                lp1 *= (1.0 - lone_shield_OH)
            else:
                lp2 *= (1.0 - lone_shield_OH)
        is_NH = (a1 == 'N' and a2 == 'H') or (a1 == 'H' and a2 == 'N')
        if is_NH:
            if a1 == 'N':
                lp1 *= (1.0 - amine_lone_shift)
            else:
                lp2 *= (1.0 - amine_lone_shift)
        lone_shield = 1.0 - lambda_lone * (lp1 + lp2) / 2.0
        if lone_shield < 0.2: lone_shield = 0.2
        Jij = J0 * order * polar_factor * ionic_factor * length_factor * sat * lone_shield
        if (i, j) in ether_pairs or (j, i) in ether_pairs:
            Jij *= (1.0 - ether_weaken)
        if (a1 == 'H' and a2 in ('O', 'N', 'F')) or (a2 == 'H' and a1 in ('O', 'N', 'F')):
            Jij += hbond_strength * order
        if (a1 == 'O' and a2 == 'O') or (a1 == 'N' and a2 == 'N'):
            Jij += mr_strength * order
        D_e += Jij
    ring_count = m - n + 1
    if ring_count > 0:
        D_e += ring_stability * ring_count
    if n > 1:
        D_e += finite_size_corr * math.log(n)
    if arom_bonds >= 6:
        hetero_factor = 1.0 - hetero_arom_decay * min(hetero_count, 5) / max(arom_bonds, 1)
        D_e += arom_strength * arom_bonds * max(hetero_factor, 0.0)
    chi_vals = [ATOM_PARAMS[a]['chi'] for a in atoms if a != 'H']
    if chi_vals:
        mean_chi = sum(chi_vals) / len(chi_vals)
        var_chi = sum((c - mean_chi) ** 2 for c in chi_vals) / len(chi_vals)
        D_e *= 1.0 + func_coeff * var_chi
    heavy_count = sum(1 for a in atoms if a != 'H')
    D_e += disp_coeff * heavy_count * (heavy_count - 1) / 2.0
    return D_e


def nn_predict(features, nn_params):
    n_features = N_FEATURES
    h = HIDDEN
    w1_end = h * n_features
    W1 = nn_params[:w1_end].reshape(h, n_features)
    offset = w1_end
    b1 = nn_params[offset:offset + h]
    offset += h
    W2 = nn_params[offset:offset + h].reshape(1, h)
    offset += h
    b2 = nn_params[offset:offset + 1]
    a1 = np.tanh(np.dot(W1, features) + b1)
    return float(np.dot(W2, a1) + b2[0])

def total_energy(atoms, bonds, phys_params, nn_params, features):
    return daolian_energy(atoms, bonds, phys_params) + nn_predict(features, nn_params)


def save_model(phys_params, nn_params, feat_min, feat_max, filepath):
    data = {
        'phys_params': phys_params.tolist(),
        'nn_params': nn_params.tolist(),
        'feat_min': feat_min.tolist(),
        'feat_max': feat_max.tolist()
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"模型已保存至 {filepath}")

def load_model(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return (np.array(data['phys_params']), np.array(data['nn_params']),
            np.array(data['feat_min']), np.array(data['feat_max']))


def load_molecules(source):
    if source == 'builtin':
        return BUILTIN_MOLECULES[:]
    elif os.path.exists(source):
        mols = []
        with open(source, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row or row[0].startswith('#'):
                    continue
                name = row[0].strip()
                atoms = row[1].strip().split('-')
                bonds_str = row[2].strip().split(';')
                bonds = []
                for b in bonds_str:
                    b = b.strip()
                    if not b:
                        continue
                    parts = b.split('-')

                    if len(parts) < 2 or len(parts) > 3:
                        print(f"警告：分子{name}中键格式错误，跳过: {b}")
                        continue
                    try:
                        i = int(parts[0])
                        j = int(parts[1])
                        order = float(parts[2]) if len(parts) == 3 else 1.0
                    except ValueError:
                        print(f"警告：分子{name}中键索引或键级无效，跳过: {b}")
                        continue

                    n = len(atoms)

                    if i < 0 and j < 0:
                        print(f"警告：分子{name}中键索引均为负，跳过: {b}")
                        continue

                    if i >= n:
                        i -= 1
                    if j >= n:
                        j -= 1

                    if i < 0 or i >= n or j < 0 or j >= n:
                        print(f"警告：分子{name}中键索引越界(原子数{n}): {b}，已跳过")
                        continue
                    bonds.append((i, j, order))
                exp = float(row[3])
                mols.append((name, atoms, bonds, exp))
        return mols
    else:
        raise FileNotFoundError(f"找不到分子数据文件: {source}")


def train_model(train_mols, model_file, max_iter_phys=500, target_mad=1.5, patience=None):

    features_list = [extract_features(atoms, bonds) for _, atoms, bonds, _ in train_mols]
    features_array = np.array(features_list)
    feat_min = features_array.min(axis=0)
    feat_max = features_array.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range==0] = 1.0
    features_norm = (features_array - feat_min) / feat_range


    print("第一阶段：优化物理参数...")
    phys_init = DEFAULT_PHYS_PARAMS.copy()
    best_mad = float('inf')
    best_phys = phys_init.copy()
    stale_steps = 0

    def callback_phys(xk, convergence):
        nonlocal best_mad, best_phys, stale_steps
        mad = 0.0
        for mol, feat in zip(train_mols, features_norm):
            D_phys = daolian_energy(mol[1], mol[2], xk)
            mad += abs(D_phys - mol[3])
        mad /= len(train_mols)
        if mad < best_mad - 0.001:
            best_mad = mad
            best_phys = xk.copy()
            stale_steps = 0
        else:
            stale_steps += 1
        if patience and stale_steps >= patience:
            return True
        if target_mad and best_mad <= target_mad:
            return True
        return False

    result = differential_evolution(
        lambda p: np.mean([abs(daolian_energy(mol[1], mol[2], p) - mol[3]) for mol in train_mols]),
        PHYS_BOUNDS, maxiter=max_iter_phys, popsize=40, tol=1e-6, disp=True, polish=False,
        callback=callback_phys, seed=123
    )
    phys_opt = best_phys
    phys_mad = best_mad
    print(f"物理模型 MAD: {phys_mad:.4f} eV")


    residuals = []
    for mol, feat in zip(train_mols, features_norm):
        D_phys = daolian_energy(mol[1], mol[2], phys_opt)
        residuals.append(mol[3] - D_phys)
    residuals = np.array(residuals)


    print("第二阶段：训练神经网络残差模型...")
    nn_best = np.random.randn(NN_PARAM_COUNT) * 0.1
    best_nn_mad = float('inf')
    lr = 0.001
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    m, v = np.zeros_like(nn_best), np.zeros_like(nn_best)
    for epoch in range(5000):
        indices = np.random.permutation(len(train_mols))
        total_loss = 0.0
        for idx in indices:
            feat = features_norm[idx]
            target = residuals[idx]

            W1 = nn_best[:HIDDEN*N_FEATURES].reshape(HIDDEN, N_FEATURES)
            offset = HIDDEN*N_FEATURES
            b1 = nn_best[offset:offset+HIDDEN]
            offset += HIDDEN
            W2 = nn_best[offset:offset+HIDDEN].reshape(1, HIDDEN)
            offset += HIDDEN
            b2 = nn_best[offset:offset+1]
            a1 = np.tanh(np.dot(W1, feat) + b1)
            pred = np.dot(W2, a1) + b2[0]
            error = pred - target

            da1 = (1 - a1**2) * np.dot(W2.T, error)
            dW2 = error * a1
            db2 = error
            dW1 = np.outer(da1, feat)
            db1 = da1
            grad = np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])

            m = beta1 * m + (1-beta1) * grad
            v = beta2 * v + (1-beta2) * grad**2
            m_hat = m / (1 - beta1**(epoch+1))
            v_hat = v / (1 - beta2**(epoch+1))
            nn_best -= lr * m_hat / (np.sqrt(v_hat) + eps)
            total_loss += error**2

        mad = 0.0
        for mol, feat, res in zip(train_mols, features_norm, residuals):
            pred_res = nn_predict(feat, nn_best)
            mad += abs(res - pred_res)
        mad /= len(train_mols)
        if mad < best_nn_mad:
            best_nn_mad = mad
            nn_opt = nn_best.copy()
        if epoch % 500 == 0:
            print(f"  神经网络 epoch {epoch}, 残差 MAD: {mad:.4f} eV")
    print(f"神经网络残差 MAD: {best_nn_mad:.4f} eV")


    final_mad = 0.0
    for mol, feat in zip(train_mols, features_norm):
        D_total = total_energy(mol[1], mol[2], phys_opt, nn_opt, feat)
        final_mad += abs(D_total - mol[3])
    final_mad /= len(train_mols)
    print(f"最终联合 MAD: {final_mad:.4f} eV")

    save_model(phys_opt, nn_opt, feat_min, feat_max, model_file)
    return phys_opt, nn_opt, feat_min, feat_max, final_mad


def main():
    parser = argparse.ArgumentParser(description='道涟理论分子能量预测模型 (最终高精度版)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--train', type=str, default='builtin')
    parser.add_argument('--test', type=str, default='builtin')
    parser.add_argument('--model', type=str, default='daolian_final_model.json')
    parser.add_argument('--max_iter', type=int, default=1000, help='物理优化最大迭代步数')
    parser.add_argument('--target_mad', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=None)
    args = parser.parse_args()

    if args.mode == 'test':
        test_mols = load_molecules(args.test)
        print(f"测试集: {len(test_mols)} 个分子")
        if os.path.exists(args.model):
            phys_params, nn_params, feat_min, feat_max = load_model(args.model)
        else:
            print("警告：未找到模型，使用默认参数（神经网络置零）")
            phys_params = DEFAULT_PHYS_PARAMS.copy()
            nn_params = np.zeros(NN_PARAM_COUNT)
            train_mols_for_norm = load_molecules('builtin')
            feats = [extract_features(atoms, bonds) for _, atoms, bonds, _ in train_mols_for_norm]
            feats = np.array(feats); feat_min = feats.min(axis=0); feat_max = feats.max(axis=0)
        test_features = [extract_features(atoms, bonds) for _, atoms, bonds, _ in test_mols]
        feat_range = feat_max - feat_min; feat_range[feat_range==0] = 1.0
        test_features_norm = [(f - feat_min) / feat_range for f in test_features]
        print(f"\n{'分子':<8} {'D_e':>8} {'实验':>8} {'偏差':>9}")
        test_mad = 0.0
        for i, mol in enumerate(test_mols):
            name, atoms, bonds, exp = mol
            D_e = total_energy(atoms, bonds, phys_params, nn_params, test_features_norm[i])
            dev = D_e - exp; test_mad += abs(dev)
            print(f"{name:<8} {D_e:8.2f} {exp:8.2f} {dev:+9.3f}")
        test_mad /= len(test_mols)
        print(f"\n测试集 MAD = {test_mad:.4f} eV"); return


    train_mols = load_molecules(args.train)
    print(f"训练集: {len(train_mols)} 个分子")
    phys_opt, nn_opt, feat_min, feat_max, train_mad = train_model(
        train_mols, args.model, args.max_iter, args.target_mad, args.patience
    )

    test_mols = load_molecules(args.test)
    test_features = [extract_features(atoms, bonds) for _, atoms, bonds, _ in test_mols]
    feat_range = feat_max - feat_min; feat_range[feat_range==0] = 1.0
    test_features_norm = [(f - feat_min) / feat_range for f in test_features]
    print(f"\n测试集结果：")
    print(f"{'分子':<8} {'D_e':>8} {'实验':>8} {'偏差':>9}")
    test_mad = 0.0
    for i, mol in enumerate(test_mols):
        name, atoms, bonds, exp = mol
        D_e = total_energy(atoms, bonds, phys_opt, nn_opt, test_features_norm[i])
        dev = D_e - exp; test_mad += abs(dev)
        print(f"{name:<8} {D_e:8.2f} {exp:8.2f} {dev:+9.3f}")
    test_mad /= len(test_mols)
    print(f"\n测试集 MAD = {test_mad:.4f} eV，训练集 MAD = {train_mad:.4f} eV")

if __name__ == "__main__":
    main()
