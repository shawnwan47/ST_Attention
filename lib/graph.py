import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import linalg
import networkx as nx

################################################################################
# Build
################################################################################

def build_graph_node(node, node_size=None, node_color='red'):
    for col in ['id', 'latitude', 'longitude']:
        assert col in node.columns
    G = nx.DiGraph()
    G.add_nodes_from(node.index)
    G.pos = {i: (tup.latitude, tup.longitude)
             for i, tup in node.iterrows()}
    G.node_size = node_size
    G.node_color = node_color
    return G


def add_road_edges(G, link):
    roads = [(i, j) for _, i, j, c in link.itertuples()
             if i in G.nodes() and j in G.nodes()]
    G.roads = roads
    return G


def add_sparse_edges(G, link):
    edges = [(u, v, w) for (u, v), w in link.iteritems()]
    G.add_weighted_edges_from(edges)
    return G


def add_dense_edges(G, adj):
    assert adj.ndim == 2
    aeq(adj.shape[0], adj.shape[1], len(G.nodes()))
    num = adj.shape[0]
    weighted_edges = [(i, j, adj[i, j])
                      for i in range(num) for j in range(num)
                      if adj[i, j] > 0]
    G.add_weighted_edges_from(weighted_edges)
    return G

################################################################################
# Draw
################################################################################

def draw_nodes(g, **kwargs):
    if g.node_size is not None:
        node_size = g.node_size / max(g.node_size) * 100
    else:
        node_size = None
    nx.draw_networkx_nodes(
        g,
        pos=g.pos,
        node_size=node_size,
        node_color=g.node_color,
        cmap=plt.get_cmap('Reds'),
        linewidths=0,
        **kwargs)


def draw_roads(g, **kwargs):
    nx.draw_networkx_edges(
        G=g,
        edgeslist=g.roads,
        pos=g.pos,
        edge_color='grey',
        alpha=0.5,
        width=1,
        arrows=False,
        edge_vmin=0,
        **kwargs)


def draw_edges(g, **kwargs):
    edges_weight = [g.edges[u, v]['weight'] for u, v in g.edges]
    nx.draw_networkx_edges(
        G=g,
        pos=g.pos,
        edge_color=edges_weight,
        edge_cmap=plt.get_cmap('Blues'),
        alpha=0.1,
        width=1,
        arrows=False,
        edge_vmin=0,
        **kwargs)


def draw_edges_of_node(g, node, **kwargs):
    edgelist = [(u, v) for u, v in g.edges if u == node]
    edge_color = [g.edges[u, v]['weight'] for u, v in edgelist]
    nx.draw_networkx_edges(
        G=g,
        pos=g.pos,
        edgelist=edgelist,
        edge_color=edge_color,
        edge_cmap=plt.get_cmap('Blues'),
        alpha=0.5,
        width=1,
        edge_vmin=0,
        edge_vmax=0.5,
        arrows=False,
        **kwargs
    )



################################################################################
# Compute
################################################################################
def digitize_dist(dist, num=16):
    dist_median = np.median(dist)
    dist[dist > dist_median] = dist_median
    dist = np.ceil(dist / dist_median * (num - 1)).astype(int)
    return dist


def graph_dist(G):
    length = nx.shortest_path_length(G, weight='weight')
    dist = {src: tgt for src, tgt in length}
    return pd.DataFrame(dist)


def graph_hop(G):
    length = nx.shortest_path_length(G)
    dist = {src: tgt for src, tgt in length}
    return pd.DataFrame(dist)


def calculate_dist_adj(dist, param=0.05):
    adj = np.exp(-np.square(dist / dist[dist < np.median(dist)].std()))
    adj[adj < param] = 0
    return adj


def calculate_od_adj(od):
    do_ = od / od.sum(0)
    od_ = od.T / od.sum(1)
    return od_.T, do_.T


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)
