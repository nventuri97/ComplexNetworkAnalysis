from igraph import *
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from random import *
from powerlaw import *

gp_graph=read("./soc-google-plus.txt", format="ncol", directed=False)
summary(gp_graph, verbosity=1, max_rows = 25, edge_list_format = 'edgelist')

vertexes=gp_graph.vs()
N=len(vertexes)
print("The number of vertexes of the graph is ", N)
logN=np.log(N)

#Take the clusters from the graph and their size
gp_clust=gp_graph.clusters(mode="WEAK")
single_cl=gp_clust.sizes()

#To translate much greater than we chose logN*10^3
def findGC(n):
    if n > logN*1000:
        return 1
    else:
        return 0

response=list(map(findGC, single_cl))
#Select the index of the element 1 and it's the index of the GC inside the cluster array
gc_index=response.index(1)
gp_GC=gp_clust.subgraph(gc_index)

summary(gp_GC, verbosity=1, max_rows = 25, edge_list_format = 'edgelist')