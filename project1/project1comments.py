import sys
import pandas as pd
import numpy as np
import networkx as nx
from scipy.special import loggamma
#from itertools import permutations
import random
from networkx.drawing.nx_pydot import write_dot

def write_gph(G, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in G.edges(): 
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def plot_gph(dag, idx2names, filename):
    named_dag = nx.relabel_nodes(dag, idx2names)
    write_dot(named_dag, filename)

def sub2ind(sizes, subscripts):
    """Convert multi-dimensional subscripts (1-based) to linear index (1-based)."""
    # Julia uses 1-based indexing; Python is 0-based
    zero_based = np.array(subscripts) - 1
    return np.ravel_multi_index(zero_based, sizes) + 1  # back to 1-based

def bayesian_score_component(M, alpha):
    """Bayesian score for one variable given parent configuration counts."""
    p = np.sum(loggamma(alpha + M)) - np.sum(loggamma(alpha)) + np.sum(loggamma(np.sum(alpha, axis=1))) - np.sum(loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    return p

def compute_counts(data, var, parents, r):
    """Vectorized computation of frequency table M_i (q_i × r_i)."""
    r_i = r[var]
    if parents:
        parent_card = [r[p] for p in parents]
        q_i = int(np.prod(parent_card))

        # Convert parent assignments to linear indices (0-based)
        parent_vals = data[:, parents] - 1  # convert to 0-based
        lin_idx = np.ravel_multi_index(parent_vals.T, parent_card)

        # Combine (parent_config, var_value)
        var_vals = data[:, var] - 1
        flat_idx = lin_idx * r_i + var_vals  # unique index per parent+child

        # Count frequencies efficiently
        counts = np.bincount(flat_idx, minlength=q_i * r_i)
        M_i = counts.reshape(q_i, r_i)
    else:
        q_i = 1
        var_vals = data[:, var] - 1
        counts = np.bincount(var_vals, minlength=r_i)
        M_i = counts[np.newaxis, :]
    return M_i

def local_score(data, var, parents, r):
    """Compute local Bayesian score for variable var given parents."""
    M_i = compute_counts(data, var, parents, r)
    alpha_i = np.ones_like(M_i)
    return bayesian_score_component(M_i, alpha_i)

def k2_algorithm(data, r, node_order, max_parents=3):
    """Run K2 using local scoring."""
    nvars = len(node_order)
    G = nx.DiGraph()
    G.add_nodes_from(range(nvars))

    local_cache = {}
    for i, var in enumerate(node_order):
        parents = []
        best_local = local_score(data, var, parents, r)
        candidates = node_order[:i]  # only earlier nodes

        improved = True
        while improved and len(parents) < max_parents:
            improved = False
            best_candidate = None
            for cand in candidates:
                if cand in parents:
                    continue
                new_parents = parents + [cand]
                key = (var, tuple(sorted(new_parents)))
                if key not in local_cache:
                    local_cache[key] = local_score(data, var, new_parents, r)
                new_score = local_cache[key]
                if new_score > best_local:
                    best_local = new_score
                    best_candidate = cand
            if best_candidate is not None:
                parents.append(best_candidate)
                G.add_edge(best_candidate, var)
                improved = True

    # total score (sum of local scores)
    total_score = sum(local_score(data, v, list(G.predecessors(v)), r) for v in range(nvars))
    return G, total_score

def compute(infile, outfile, n_permutations=5, max_parents=6):
    df = pd.read_csv(infile)
    data = df.to_numpy(dtype=int)
    nvars = data.shape[1]
    idx2names = list(df.columns)
    r = [int(np.max(data[:, i])) for i in range(nvars)]
    #print("r: ", r)
    
    order = list(range(nvars))
    random.seed(238)
    best_score = float('-inf')
    best_G = None

    for p in range(n_permutations):        
        random.shuffle(order)
        G, score = k2_algorithm(data, r, order, max_parents=max_parents)
        print("Permutation ", p, "score= ", score)
        if score > best_score:
            best_score = score
            best_G = G.copy()

    write_gph(best_G, idx2names, outfile)
    print("Best score ", best_score)

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")
    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)

if __name__ == "__main__":
    main()
