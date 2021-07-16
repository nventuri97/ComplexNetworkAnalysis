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

#-----Erdos Renyi graph comparison-----
er_gp_all = Graph.Erdos_Renyi(n = gp_GC.vcount(), m = gp_GC.ecount())

# take only the Giant Component
er_gp = er_gp_all.clusters(mode = "WEAK").giant()
print("The number of vertexes of giant component's Erdos Renyi G+ graph ",er_gp.vcount())
print("The number of vertexes of giant component's the original G+ graph ",gp_GC.vcount())

# we use GridSpecs for a finer control of the plot positioning
fig_sizes = (fig_sizes[0], 2*default_sizes[1])
f = plt.figure(figsize = fig_sizes)

# create a 2x2 Grid Specification
gs = gridspec.GridSpec(2, 2)

# add subplots to the figure, using the GridSpec gs
# position [0,0] (upper-left corner)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
# the third plot spans the entire second row
ax3 = plt.subplot(gs[1,:])

# compute and plot the histogram of FB degrees
d_gp = gp_GC.degree()
_,_,_ = ax1.hist(d_gp, bins=range(1,max(d_gp)+2), density = True, color = 'red')
_ = ax1.set_xlim(0,80)
_ = ax1.set_xlabel("$d$")
_ = ax1.set_ylabel("Frequencies")
_ = ax1.set_title("ISTOGRAMMA GRADI GPLUS")

# compute and plot the histogram of ER degrees
d_er = er_gp.degree()
_,_,_ = ax2.hist(d_er, bins=range(1,max(d_er)+2), density = True, color = 'blue')
_ = ax2.set_xlim(0,80)
_ = ax2.set_xlabel("$d$")
_ = ax2.set_ylabel("Frequencies")
_ = ax2.set_title("ISTOGRAMMA GRADI ER")

# compute and plot the degree CCDFs
gp_ecdf = ECDF(d_gp)
er_ecdf = ECDF(d_er)
x = np.arange(1,max(d_gp)+1)
_ = ax3.loglog(x, 1-gp_ecdf(x), 'ro', label = 'GPLUS')
x = np.arange(1,max(d_er)+1)
_ = ax3.loglog(x, 1-er_ecdf(x), 'bo', label = 'Erdos-Renyi')
_ = ax3.set_xlabel("$d$")
_ = ax3.set_ylabel("$P(D>d)$")
_ = ax3.set_title("Comparison between degree CCDFs")
_ = ax3.legend(numpoints = 1)

plt.show()

# clustering
print("Giant component's global transitivity Erdos Renyi graph ",er_gp.transitivity_undirected())
print("Giant component's local transitivity Erdos Renyi graph for nodes with less then 2 neighbours ",er_gp.transitivity_avglocal_undirected(mode="zero"))

# Shortest path lenght
# on a subset of the nodes, as otherwise it will take forever to compute
gp_vs_src = sample(list(gp_GC.vs), 1000)
gp_vs_trg = sample(list(gp_GC.vs), 1000)
er_vs_src = sample(list(er_gp.vs), 1000)
er_vs_trg = sample(list(er_gp.vs), 1000)

gp_sp = mean(np.array(gp_GC.shortest_paths(gp_vs_src, gp_vs_trg, weights=None)).flatten())
print("Erdos Renyi shortest path avg ",mean(np.array(er_gp.shortest_paths(er_vs_src, er_vs_trg)).flatten()))

#-----Fixed power-law graph comparison-----
# Analyse the entire graph

# First, we find the best power law fit for the degree distribution
# with a fixed minimum value xmin (minimum degree for which the fitting is computed)
# - see the plot of the CCDFs for understanding how fitting depends on xmin
xmin = 100
fit_pl = Fit(gp_GC.degree(), xmin = xmin)
# by computing automatically the "best" xmin value
fit_pl_auto = Fit(gp_GC.degree())

exp_pl_auto = fit_pl_auto.alpha
xmin_auto = fit_pl_auto.xmin
exp_pl = fit_pl.alpha
print ("PL exponents: (xmin=%d) %.2f; (auto xmin=%.2f) %.2f" % (xmin, exp_pl, xmin_auto, exp_pl_auto))

# compute the number of nodes and edges of the FB graph to generate the equivalent static Power Law graph
N = gp_GC.vcount()
M = gp_GC.ecount()

# Equivalent graph for the fitting with fixed xmin
pl_gp_all = Graph.Static_Power_Law(N, M, exp_pl)
# the graph could not be connected, so keep the GC only
pl_gp = pl_gp_all.clusters(mode = "WEAK").giant()

# Equivalent graph for the fitting with automatic xmin
pl_gp_auto_all = Graph.Static_Power_Law(N, M, exp_pl_auto)
pl_gp_auto = pl_gp_auto_all.clusters(mode = "WEAK").giant()

# clustering coefficients
cc_pl = pl_gp.transitivity_undirected()
cc_pl_auto = pl_gp_auto.transitivity_undirected()
cc_gp = gp_GC.transitivity_undirected()
print ("Clustering: (GP) %.5f; (PL xmin=%d) %.5f; (PL auto xmin) %.5f" %(cc_gp, xmin, cc_pl, cc_pl_auto))

# Shortest path lenght
# on a subset of the nodes, as otherwise it will take forever to compute
pl_vs_src = sample(list(pl_gp.vs), 500)
pl_vs_trg = sample(list(pl_gp.vs), 500)

pl_auto_vs_src = sample(list(pl_gp_auto.vs), 500)
pl_auto_vs_trg = sample(list(pl_gp_auto.vs), 500)

sp_pl = mean(np.array(pl_gp.shortest_paths(pl_vs_src, pl_vs_trg)).flatten())
sp_pl_auto = mean(np.array(pl_gp_auto.shortest_paths(pl_auto_vs_src, pl_auto_vs_trg)).flatten())
print ("Shortest paths: (GP) %.2f; (PL xmin=%d) %.2f; (PL auto xmin) %.2f" % (gp_sp, xmin, sp_pl, sp_pl_auto))

# Now we compare the degree distributions for the complete GP Giant Component
# we use GridSpecs for a finer control of the plot positioning
fig_sizes = (fig_sizes[0], 2*default_sizes[1])
f = plt.figure(figsize = fig_sizes)

# create a 2x3 Grid Specification
gs = gridspec.GridSpec(2, 3)

# add subplots to the figure, using the GridSpec gs
# position [0,0] (upper-left corner)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])
# the fourth plot spans the entire second row
ax4 = plt.subplot(gs[1,:])

# compute and plot the histogram of FB degrees
d_gp = gp_GC.degree()
_,_,_ = ax1.hist(d_gp, bins=range(1,max(d_gp)+2), density = True, color = 'red')
_ = ax1.set_xlim(0,80)
_ = ax1.set_xlabel("$d$")
_ = ax1.set_ylabel("Frequencies")
_ = ax1.set_title("Histogram of GP degrees")

# compute and plot the histogram of Static Power Law degrees with set xmin
d_pl = pl_gp.degree()
_,_,_ = ax2.hist(d_pl, bins=range(1,max(d_pl)+2), density = True, color = 'blue', label = "$\gamma$ = %.2f" % exp_pl)
_ = ax2.set_xlim(0,80)
_ = ax2.set_xlabel("$d$")
_ = ax2.set_ylabel("Frequencies")
_ = ax2.set_title("Histogram of PL degrees (xmin=%d)" % xmin)
_ = ax2.legend()

# compute and plot the histogram of Static Power law degrees with auto xmin
d_pl_auto = pl_gp_auto.degree()
_,_,_ = ax3.hist(d_pl_auto, bins=range(1,max(d_pl_auto)+2), density = True, color = 'green', label = "$\gamma$ = %.2f" % exp_pl_auto)
_ = ax3.set_xlim(0,80)
_ = ax3.set_xlabel("$d$")
_ = ax3.set_ylabel("Frequencies")
_ = ax3.set_title("Histogram of PL degrees (auto xmin)")
_ = ax3.legend()

# compute and plot the degree CCDFs
gp_ecdf = ECDF(d_gp)
pl_ecdf = ECDF(d_pl)
pl_auto_ecdf = ECDF(d_pl_auto)
x = np.arange(1,max(d_gp)+1)
_ = ax4.loglog(x, 1-gp_ecdf(x), 'ro', label = 'GPLUS')
x = np.arange(1,max(d_pl)+1)
_ = ax4.loglog(x, 1-pl_ecdf(x), 'bo', label = 'Static PL xmin=%d' % xmin)
x = np.arange(1,max(d_pl_auto)+1)
_ = ax4.loglog(x, 1-pl_auto_ecdf(x), 'go', label = 'Static PL auto xmin')
_ = ax4.set_xlabel("$d$")
_ = ax4.set_ylabel("$P(D>d)$")
_ = ax4.set_title("Comparison between degree CCDFs")
_ = ax4.legend(numpoints = 1)

# for reference, plot the power law functions corresponding to the fitting with fixed and automatic xmin
x1 = np.arange(xmin_auto, max(d_gp)+1)
_ = ax4.loglog(x1, 1000000000000 * x1**(-exp_pl_auto), 'g-', linewidth = 3)
x1 = np.arange(xmin, max(d_gp)+1)
_ = ax4.loglog(x1, 1000 * x1**(-exp_pl), 'b-', linewidth = 2)

plt.show()