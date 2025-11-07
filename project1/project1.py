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

def compute_Mi(data, var, parents, r): # julia statistics, computes the counts, M_i 
    #Mi[j,k]=Nijk - counts of parent config j where Xi=k
    #q_i = number of parent configurations of variable i
    #r_i = n possible values for var i
    r_i = r[var]
    # print("compute Mi parents : ", parents)
    # print("var : ", var)
    if not parents: # 1 parent config
        q_i = 1 # Mi-> size (1, r_i)
        var_vals = data[:, var] - 1
        counts = np.bincount(var_vals)
        M_i = counts[np.newaxis, :]     
    else:
        n_parent_config = [r[p] for p in parents]  # num of possible parent configs
        q_i = int(np.prod(n_parent_config))
        parent_vals = data[:, parents] - 1  # 0-based
        lin_idx = np.ravel_multi_index(parent_vals.T, n_parent_config) #tuple of index arrays -> array of flat indices
        # combine parent_config, var_value
        var_vals = data[:, var] - 1
        flat_idx = lin_idx * r_i + var_vals  # flat idx for parent+child
        counts = np.bincount(flat_idx, minlength=q_i*r_i) #flat counts
        M_i = counts.reshape(q_i, r_i)
    #print(M_i)
    return M_i

def bayesian_score_component(M, a):
    p = np.sum(loggamma(a + M)) 
    p -= np.sum(loggamma(a)) 
    p += np.sum(loggamma(np.sum(a, axis=1))) 
    p -= np.sum(loggamma(np.sum(a, axis=1) + np.sum(M, axis=1)))
    return p
# local score for a var given its parents
def local_bayesian_score(data, var, parents, r):
    #M[i] = counts table of size qi x ri 
    M_i = compute_Mi(data, var, parents, r) 
    a_i = np.ones_like(M_i) #a[i] = uniform Dirichlet prior, same size as Mi
    return bayesian_score_component(M_i, a_i) #computes local contribution for node i

def k2_algorithm(data, var_order, r, max_parents):
    n_var = len(var_order)
    G = nx.DiGraph()
    G.add_nodes_from(range(n_var))
    local_cache = {}
    for i, node in enumerate(var_order): # in order
        parents = [] # start w no parents
        curscore = local_bayesian_score(data, node, parents, r)
        while len(parents) < max_parents: # within max parents
            bestparent = None
            bestscore = curscore
            for par in var_order[:i]: # potential parents r only nodes before cur node
                if par in parents: 
                    continue
                # check new score
                new_parents = parents + [par]
                key = (node, tuple(sorted(new_parents)))
                if key not in local_cache:
                    local_cache[key] = local_bayesian_score(data, node, new_parents, r)
                newscore = local_cache[key]
                # print("potential parent: ", potpar, " score: ", score, "best score", bestscore)
                if newscore > bestscore: 
                    bestscore = newscore
                    bestparent = par
            if bestscore>curscore: # add parent w/ best score in var_order[:i]
                parents.append(bestparent)
                G.add_edge(bestparent, node) # parent -> cur node
                curscore = bestscore
                #print(current_score)
            else:
                break
    #sum of local scores
    total_score = sum(local_bayesian_score(data, v, list(G.predecessors(v)), r) for v in range(n_var))
    return G, total_score

def compute(infile, outfile, n_permutations):
    df = pd.read_csv(infile)
    data = df.to_numpy(dtype=int)
    nvars = data.shape[1]
    idx2names = list(df.columns) 
    r = [int(np.max(data[:, i])) for i in range(nvars)] # n states every var can take on
    #print("r: ", r)
    
    order = list(range(nvars))
    random.seed(238)
    best_score = float('-inf')
    best_G = None

    for p in range(n_permutations):        
        random.shuffle(order)
        G, score = k2_algorithm(data, order, r, max_parents=3)
        print("Permutation ", p, "score= ", score)
        if score > best_score:
            best_score = score
            best_G = G.copy()

    write_gph(best_G, idx2names, outfile)
    # plot_gph(G, idx2names, outfile) #.dot
    print("Best score ", best_score)

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")
    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename, n_permutations=5)

if __name__ == "__main__":
    main()
