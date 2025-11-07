import sys
import networkx
import pandas as pd
import numpy as np
from scipy.special import loggamma
from networkx.drawing.nx_pydot import write_dot

def write_gph(G, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in G.edges(): 
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def plot_gph(dag, idx2names, filename):
    named_dag = networkx.relabel_nodes(dag, idx2names)
    write_dot(named_dag, filename)

# local k2 bayesian score for a single node given its parents 
# uniform Dirichlet prior (α_ijk = 1) 
def score(data, node, parents): 
    N = len(data)
    score = 0.0
    r_i = max(data[:, node]) + 1  # number of possible values for node i
    # No parents -> compute marginal distribution
    if not parents: # qi = 1 parent config, Nij=Ni= total samples
        counts = np.bincount(data[:, node]) #Nijk =Nik for each Xi=k
        # counts = [np.sum(data[:, node] == k) for k in range(r_i)] 
        score = loggamma(r_i) - loggamma(N + r_i)
        for c in counts:
            score += loggamma(c+1)
    else: # Has parents
        parent_configs = {} # get list
        for sample in data: 
            parent_config = tuple(sample[p] for p in parents) # unique config:list of samples
            if parent_config in parent_configs:
                parent_configs[parent_config].append(sample[node])
            else:
                parent_configs[parent_config]=[]
        for Xvalues in parent_configs.values(): # list of X samples for each parent config
            N_ijk_Counts = np.bincount(Xvalues) # num of samples w/ cur parent config j & Xi=k
            N_ij = len(Xvalues) # num of samples w/ cur parent config
            score += loggamma(r_i) - loggamma(N_ij + r_i)
            for N_ijk in N_ijk_Counts:
                score += loggamma(N_ijk + 1)
    return score

def k2_algorithm(data, var_order, max_parents):
    G = networkx.DiGraph()
    G.add_nodes_from(range(data.shape[1]))
    for i, node in enumerate(var_order): # go thru order
        parents = [] # start w no parents
        curscore = score(data, node, parents)
        while len(parents) < max_parents: # within max parents
            bestparent = None
            bestscore = curscore
            for potpar in var_order[:i]: # potential parents r only nodes before cur node
                if potpar in parents: 
                    continue
                # check new score
                newscore = score(data, node, parents + [potpar])
                # print("potential parent: ", potpar, " score: ", score, "best score", bestscore)
                if newscore > bestscore: 
                    bestscore = newscore
                    bestparent = potpar
            if bestscore>curscore: # add parent w/ best score in var_order[:i]
                parents.append(bestparent)
                G.add_edge(bestparent, node) # parent -> cur node
                curscore = bestscore
                #print(current_score)
            else:
                break
    return G

def total_score(data, G):
    total = 0.0
    for node in G.nodes():
        parents = list(G.predecessors(node))
        total += score(data, node, parents)
    print("Total score: ", total)

def compute(infile, outfile):
    # read csv
    df = pd.read_csv(infile)
    data = df.to_numpy(dtype=int)    
    variables = list(df.columns)
    
    #all_permutations = permutations(variables) # diff initial orders
    # use variable order
    var_order = list(range(len(variables))) 
    G = k2_algorithm(data, var_order, max_parents=3)
    
    # output
    total_score(data, G)
    idx2names = {i: name for i, name in enumerate(variables)}
    write_gph(G, idx2names, outfile)
    # plot_gph(G, idx2names, outfile) #.dot

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)

if __name__ == '__main__':
    main()
