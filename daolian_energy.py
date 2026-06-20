import math, csv, json, os, sys, argparse
import numpy as np
from scipy.optimize import differential_evolution
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

ATOM_PARAMS = {
    'H': {'nf': 1, 'chi': 2.20, 'r_cov': 0.31, 'n_valence': 1, 'has_lone': False},
    'C': {'nf': 2, 'chi': 2.55, 'r_cov': 0.76, 'n_valence': 4, 'has_lone': False},
    'N': {'nf': 2, 'chi': 3.04, 'r_cov': 0.71, 'n_valence': 4, 'has_lone': True},
    'O': {'nf': 2, 'chi': 3.44, 'r_cov': 0.66, 'n_valence': 4, 'has_lone': True},
    'F': {'nf': 2, 'chi': 3.98, 'r_cov': 0.57, 'n_valence': 4, 'has_lone': True},
    'S': {'nf': 3, 'chi': 2.58, 'r_cov': 1.05, 'n_valence': 6, 'has_lone': True},
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
    ('H2S',  ['S','H','H'],            [(0,1,1),(0,2,1)], 7.40),
    ('CS2',  ['C','S','S'],            [(0,1,2),(0,2,2)], 11.96),
    ('SO2',  ['S','O','O'],            [(0,1,2),(0,2,2)], 11.12),
    ('CH3SH',['C','S','H','H','H','H'], [(0,1,1),(0,2,1),(0,3,1),(0,4,1),(1,5,1)], 19.58),
]

def find_sssr(atoms, bonds, max_ring_size=10):
    n = len(atoms)
    adj = defaultdict(list)
    for b in bonds:
        i, j = b[0], b[1]
        if 0 <= i < n and 0 <= j < n:
            adj[i].append(j)
            adj[j].append(i)
    all_rings = []
    for start in range(n):
        if atoms[start] == 'H': continue
        queue = deque()
        queue.append((start, [start]))
        visited = {start: [start]}
        while queue:
            curr, path = queue.popleft()
            if len(path) > max_ring_size: continue
            for nxt in adj.get(curr, []):
                if nxt == start and len(path) >= 3:
                    all_rings.append(path)
                elif nxt not in visited and atoms[nxt] != 'H':
                    visited[nxt] = path + [nxt]
                    queue.append((nxt, path + [nxt]))
    uniq_rings = []
    for r in all_rings:
        s = frozenset(r)
        if not any(s == frozenset(u) for u in uniq_rings):
            uniq_rings.append(list(r))
    return uniq_rings

def compute_aromatic_braid(atoms, bonds):
    arom_edges = []
    for bd in bonds:
        i, j = bd[0], bd[1]
        order = bd[2] if len(bd) == 3 else 1.0
        if order == 1.5 and atoms[i] == 'C' and atoms[j] == 'C':
            arom_edges.append((i, j))
    if not arom_edges:
        return 0, 0.0, set(), set()

    adj = defaultdict(list)
    for u, v in arom_edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = set()
    total_N_region = 0
    arom_set = set()
    carbons = set()

    for start in adj:
        if start in visited: continue
        comp_nodes = []
        queue = [start]
        visited.add(start)
        while queue:
            cur = queue.pop(0)
            comp_nodes.append(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        comp_edges = set()
        for u in comp_nodes:
            for v in adj[u]:
                if u < v and v in set(comp_nodes):
                    comp_edges.add((u, v))
        V = len(comp_nodes)
        E = len(comp_edges)
        if V >= 3 and E == V:
            N_region = E // 2
        elif V >= 3 and E > V:
            num_rings = E - V + 1
            avg_size = (2 * E) / num_rings if num_rings > 0 else V
            N_region = int(num_rings * (avg_size / 2))
        else:
            N_region = 0
        total_N_region += N_region
        for u, v in comp_edges:
            arom_set.add((min(u, v), max(u, v)))
        for node in comp_nodes:
            carbons.add(node)

    K_CC_unit_fixed = 4.748 * (2.0 / (2 + 2)) ** 0.028
    return total_N_region, K_CC_unit_fixed, arom_set, carbons

def daolian_energy(atoms, bonds, phys_params=None):
    delta_shell = 0.028
    base_sat_double = 0.85
    base_sat_triple = 0.72
    eta_single_CC = 0.78
    eta_single_CH = 0.92

    if phys_params is None:
        delta_Pauli = 0.60
        ring_energy_offset = 0.0
        ether_weaken = 0.0
        eta_CO_single = 1.0
        eta_CN_single = 1.0
        eta_CF_single = 1.0
        ring_strain_factor = 0.0
    else:
        (delta_Pauli, ring_energy_offset, ether_weaken,
         eta_CO_single, eta_CN_single, eta_CF_single,
         ring_strain_factor) = phys_params

    n = len(atoms)
    D_e = 0.0

    N_region_total, K_CC_unit_fixed, arom_bonds_set, arom_carbons = compute_aromatic_braid(atoms, bonds)
    arom_indices = set()
    for idx, bd in enumerate(bonds):
        i, j = bd[0], bd[1]
        if (min(i, j), max(i, j)) in arom_bonds_set:
            arom_indices.add(idx)

    degree = [0] * n
    for bd in bonds:
        degree[bd[0]] += 1
        degree[bd[1]] += 1

    lambda_atom = []
    for i, a in enumerate(atoms):
        n_bond = degree[i]
        n_val = ATOM_PARAMS[a]['n_valence']
        if ATOM_PARAMS[a].get('has_lone', False):
            lam = (n_bond / n_val) ** delta_Pauli if n_val > 0 else 1.0
        else:
            lam = 1.0
        lambda_atom.append(lam)

    for idx, bd in enumerate(bonds):
        if idx in arom_indices:
            continue
        i, j = bd[0], bd[1]
        order = bd[2] if len(bd) == 3 else 1.0
        a1, a2 = atoms[i], atoms[j]

        if order == 1.0 and ((a1 == 'C' and i in arom_carbons and a2 == 'H') or
                             (a2 == 'C' and j in arom_carbons and a1 == 'H')):
            nf_c, nf_h = 2, 1
            cov_fixed = (2.0 / (nf_c + nf_h)) ** delta_shell
            K_CH_fixed = 4.748 * cov_fixed
            D_e += K_CH_fixed
            continue

        nf1, nf2 = ATOM_PARAMS[a1]['nf'], ATOM_PARAMS[a2]['nf']
        cov_factor = (2.0 / (nf1 + nf2)) ** delta_shell
        K_single = 4.748 * cov_factor

        if order <= 1.0:
            sat_factors = [1.0]
        elif order == 2:
            sat_factors = [1.0, base_sat_double]
        elif order == 3:
            sat_factors = [1.0, base_sat_double, base_sat_triple]
        else:
            frac = order - int(order)
            sat_factors = [1.0, 1.0 - (1.0 - base_sat_double) * frac]
        pair_sum = sum(sat_factors)

        lam1, lam2 = lambda_atom[i], lambda_atom[j]
        lone_compress = math.sqrt(lam1 * lam2)

        Jeff = K_single * pair_sum * lone_compress

        if order == 1.0:
            if (a1 == 'C' and a2 == 'C'):
                Jeff *= eta_single_CC
            elif (a1 == 'C' and a2 == 'H') or (a1 == 'H' and a2 == 'C'):
                if not ((a1 == 'C' and i in arom_carbons) or (a2 == 'C' and j in arom_carbons)):
                    Jeff *= eta_single_CH
            elif (a1 == 'C' and a2 == 'O') or (a1 == 'O' and a2 == 'C'):
                Jeff *= eta_CO_single
            elif (a1 == 'C' and a2 == 'N') or (a1 == 'N' and a2 == 'C'):
                Jeff *= eta_CN_single
            elif (a1 == 'C' and a2 == 'F') or (a1 == 'F' and a2 == 'C'):
                Jeff *= eta_CF_single

        D_e += Jeff

    if N_region_total > 0:
        D_e += 2.0 * N_region_total * K_CC_unit_fixed
        D_e += ring_energy_offset * N_region_total

    rings_raw = find_sssr(atoms, bonds)
    ring_cnt = len(rings_raw)
    if ring_cnt > 0:
        D_e += ether_weaken * ring_cnt

        if ring_strain_factor != 0.0:
            for ring in rings_raw:
                if not all(atoms[v] == 'C' for v in ring): continue

                orders_in_ring = []
                for k in range(len(ring)):
                    a, b = ring[k], ring[(k+1)%len(ring)]
                    for bd in bonds:
                        if (bd[0]==a and bd[1]==b) or (bd[0]==b and bd[1]==a):
                            order = bd[2] if len(bd)==3 else 1.0
                            orders_in_ring.append(order)
                            break
                if all(o == 1.5 for o in orders_in_ring):
                    continue

                D_e += ring_strain_factor * len(ring)

    return D_e

def load_molecules(source):
    if source == 'builtin':
        return BUILTIN_MOLECULES[:]
    elif os.path.exists(source):
        mols = []
        with open(source, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row or row[0].startswith('#'): continue
                name = row[0].strip()
                atoms = row[1].strip().split('-')
                bonds_str = row[2].strip().split(';')
                bonds = []
                for b in bonds_str:
                    b = b.strip()
                    if not b: continue
                    parts = b.split('-')
                    if len(parts) < 2: continue
                    try:
                        i = int(parts[0]); j = int(parts[1])
                        order = float(parts[2]) if len(parts) == 3 else 1.0
                    except ValueError:
                        continue
                    if i >= len(atoms): i -= 1
                    if j >= len(atoms): j -= 1
                    if i < 0 or i >= len(atoms) or j < 0 or j >= len(atoms):
                        print(f"警告：分子{name}中键索引越界: {b}")
                        continue
                    bonds.append((i, j, order))
                exp = float(row[3])
                mols.append((name, atoms, bonds, exp))
        return mols
    else:
        raise FileNotFoundError(f"找不到分子数据文件: {source}")

def compute_mad(phys, mols):
    loss = 0.0
    for _, at, bd, exp in mols:
        loss += abs(daolian_energy(at, bd, phys) - exp)
    return loss / len(mols)

def train_model(train_mols, phys_init, phys_bnd, max_iter=500):
    best_mad = float('inf')
    best_params = phys_init.copy()

    def callback(xk, conv):
        nonlocal best_mad, best_params
        mad = compute_mad(xk, train_mols)
        if mad < best_mad:
            best_mad = mad
            best_params = xk.copy()
        return False

    print("优化物理参数...")
    result = differential_evolution(
        lambda p: compute_mad(p, train_mols),
        phys_bnd, maxiter=max_iter, popsize=40, tol=1e-6, disp=True, polish=False,
        callback=callback, seed=123
    )
    phys_opt = best_params
    print(f"训练集 MAD: {best_mad:.4f} eV")
    return phys_opt, best_mad

def main():
    parser = argparse.ArgumentParser(description='道涟 杂原子修正版')
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--train', default='builtin')
    parser.add_argument('--test', default='builtin')
    parser.add_argument('--model', default='daolian_hetero.json')
    parser.add_argument('--max_iter', type=int, default=200)
    args = parser.parse_args()

    DEFAULT_PHYS_PARAMS = np.array([0.60, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    PHYS_BOUNDS = [
        (0.50, 0.70),
        (-0.1, 0.1),
        (-0.05, 0.05),
        (0.5, 1.0),
        (0.5, 1.0),
        (0.5, 1.0),
        (-0.1, 0.1)
    ]

    if args.mode == 'test':
        test_mols = load_molecules(args.test)
        print(f"测试集: {len(test_mols)} 个分子")
        if os.path.exists(args.model):
            with open(args.model, 'r') as f:
                phys = np.array(json.load(f)['phys'])
        else:
            phys = DEFAULT_PHYS_PARAMS.copy()
        print(f"\n{'分子':<8} {'D_e':>8} {'实验':>8} {'偏差':>9}")
        mad = 0.0
        for name, at, bd, exp in test_mols:
            de = daolian_energy(at, bd, phys)
            dev = de - exp; mad += abs(dev)
            print(f"{name:<8} {de:8.2f} {exp:8.2f} {dev:+9.3f}")
        print(f"\n测试集 MAD = {mad/len(test_mols):.4f} eV")
        return

    train_mols = load_molecules(args.train)
    print(f"训练集: {len(train_mols)} 个分子")
    phys_opt, train_mad = train_model(train_mols, DEFAULT_PHYS_PARAMS.copy(), PHYS_BOUNDS, args.max_iter)
    with open(args.model, 'w') as f:
        json.dump({'phys': phys_opt.tolist()}, f, indent=2)
    print(f"模型已保存至 {args.model}，训练集 MAD = {train_mad:.4f} eV")

    test_mols = load_molecules(args.test)
    print(f"\n测试集结果：")
    print(f"{'分子':<8} {'D_e':>8} {'实验':>8} {'偏差':>9}")
    mad = 0.0
    for name, at, bd, exp in test_mols:
        de = daolian_energy(at, bd, phys_opt)
        dev = de - exp; mad += abs(dev)
        print(f"{name:<8} {de:8.2f} {exp:8.2f} {dev:+9.3f}")
    print(f"\n测试集 MAD = {mad/len(test_mols):.4f} eV")

if __name__ == "__main__":
    main()
