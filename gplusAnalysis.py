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
#gp_graph=read("./soc-google-plus.10K.sb.txt", format="ncol", directed=False)
#gp_graph=read("./soc-google-plus.38K.sb.txt", format="ncol", directed=False)
#gp_graph=read("./soc-google-plus.62K.sb.txt", format="ncol", directed=False)
#gp_graph=read("./soc-google-plus.115K.sb.txt", format="ncol", directed=False)
summary(gp_graph, verbosity=1, max_rows = 25, edge_list_format = 'edgelist')

vertexes=gp_graph.vs()
N=len(vertexes)
print("The number of vertexes of the graph is ", N)
logN=np.log(N)

#------Giant Component-----
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

gp_graph.vs["color"]='red'
print(len(gp_clust.membership))
print(gp_graph.vcount())
for i in range(gp_graph.vcount()):
    if gp_clust.membership[i]==gc_index:
        gp_graph.vs[i]["color"]='blue'

gp_GC=gp_clust.subgraph(gc_index)

summary(gp_GC, verbosity=1, max_rows = 25, edge_list_format = 'edgelist')

try:
    del visual_style
    visual_style = {}
except NameError:
    visual_style = {}
    
visual_style["bbox"] = (600,600)
visual_style["label"] = []
visual_style["layout"] = gp_graph.layout_fruchterman_reingold()
visual_style["vertex_size"] = 5
visual_style["vertex_shape"] = 'circle'
visual_style["edge_arrow_size"] = 0.2

plot(gp_GC, **visual_style)

#-----Degree Distribution------
gp_GC_degree= gp_GC.degree()
id_max=np.argmax(gp_GC_degree)
print("Node ", id_max," has maximum degree ",gp_GC_degree[id_max])

#GP graph is undirected graph, so the right number of neighbours is neighbours value
nei=gp_GC.neighbors(id_max)
neighbours= gp_GC.neighborhood(id_max, order=1, mode="all")
print("Node ", id_max," has ", len(neighbours)-1, " neighbours")

dd_h, dd_h_bins, _ = plt.hist(gp_GC_degree, bins=range(1,max(gp_GC_degree)+2), density=True, color = 'red')
plt.show()

#-----CCDF-----
#ECDF(dataset) returns the empirical CDF computed from the dataset
gp_deg_cdf= ECDF(gp_GC_degree)

default_sizes = plt.rcParams["figure.figsize"]
fig_sizes = (2*default_sizes[0], default_sizes[1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = fig_sizes)

degs = np.arange(1,max(gp_GC_degree)+1)
ax1.plot(degs, 1-gp_deg_cdf(degs), 'bo')
ax1.set_xlabel("$d$")
ax1.set_ylabel("$P(D>d)$")
ax1.set_title("Degree CCDF in a lin-lin scale")

ax2.loglog(degs, 1-gp_deg_cdf(degs), 'bo')
ax2.set_xlabel("$d$")
ax2.set_ylabel("$P(D>d)$")
ax2.set_title("Degree CCDF in a log-log scale")

plt.show()

#-----Assortativity-----
gp_GC_simple=gp_GC.simplify()
ass_deg=gp_GC_simple.assortativity_degree()
print("Giant component's assortativity degree ", ass_deg)

gp_knn, gp_knnk= gp_GC_simple.knn()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = fig_sizes)
ax1.loglog(gp_GC_simple.degree(), gp_knn, 'go')
ax1.set_xlabel("degree")
ax1.set_ylabel("Neighbors degree")
ax1.set_title("$knn$ index for the G+ Giant Component")

ax2.loglog(range(1,max(gp_GC_simple.degree())+1), gp_knnk, 'go')
ax2.set_xlabel("degree")
ax2.set_ylabel("Average degree of neighbors")
ax2.set_title("$knnk$ index for the G+ Giant Component")

plt.show()

#-----Local and global clustering-----
transitivity=gp_GC.transitivity_undirected()
print("Giant component's global transitivity ", transitivity)

local_trans=gp_GC.transitivity_avglocal_undirected(mode="zero")
print("Giant component's local transitivity for nodes with less then 2 neighbours", local_trans)

local_all_t=gp_GC.transitivity_local_undirected(mode="zero")
print("The mean of the Giant component's local transitivity ", mean(local_all_t))

#-----Centrality-----
#We have to use a subset of the graph for computational purpose
idx = np.argwhere(np.array(gp_GC.degree())>60).flatten()
sub_g_gp = gp_GC.induced_subgraph(idx)

cl_cent=sub_g_gp.closeness()
print("\n")
print("Closeness centrality of the first twenty nodes is:")
for i in range(0,19):
    print(cl_cent[i])

bet_cent=sub_g_gp.betweenness()
print("\n")
print("Betweenness centrality of the first twenty nodes is")
for i in range(0,19):
    print(bet_cent[i])
print("\n")

edge_bet = sub_g_gp.edge_betweenness()
idx_max = np.argmax(edge_bet)

print("Edge with maximum betweenness (", edge_bet[idx_max],") is ", sub_g_gp.es[idx_max].tuple)

#-----Community Detecton-----
idx = np.argwhere(np.array(gp_GC.degree())>100).flatten()
sub_g1 = gp_GC.induced_subgraph(idx)

vd = sub_g1.community_fastgreedy()

print("The community detected are ", vd.optimal_count)
vd_clust = vd.as_clustering()
print("The sizes of the communities are ", vd_clust.sizes())
cros = np.array(vd_clust.crossing())
edge_cross=np.argwhere(cros == True).flatten()
print("There are ", len(edge_cross), "edges that cut across between commiunity")

plot(vd_clust, layout=sub_g1.layout_fruchterman_reingold(), mark_groups = True)

#-----Unweighted path lenghts-----
src=sample(list(gp_GC.vs), 100)
trg=sample(list(gp_GC.vs), 100)

n_hops_u = gp_GC.shortest_paths(source = src, target = trg, weights = None)
n_hops_u = np.array(n_hops_u).flatten()

_,_,_ = plt.hist(n_hops_u, bins = range(1,max(n_hops_u)+2), density = True)
_ = plt.axvline(mean(n_hops_u), color = 'gold', linewidth = 4)
_ = plt.xlabel("# of hops")
_ = plt.ylabel("frequency")
_ = plt.title("Histogram of UNWEIGHTED path length")

plt.show()