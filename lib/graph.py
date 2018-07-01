import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import linalg
import networkx as nx


def build_link_graph(node, link):
    G = build_node_graph(node)
    edges = [(i, j, c) for _, i, j, c in link.itertuples()
             if i in G.nodes() and j in G.nodes()]
    G.add_weighted_edges_from(edges)
    return G


def build_node_graph(node):
    G = nx.DiGraph()
    G.add_nodes_from(node['id'])
    G.pos = {tup.id: (tup.latitude, tup.longitude)
             for _, tup in node.iterrows()}
    return G


def build_multi_attention_graph(attention, node):
    assert attention.ndim() == 3
    head, len_q, len_c = attention.shape
    Gs = (build_attention_graph(attention[i], node) for i in range(head))
    return Gs


def build_attention_graph(attention, node):
    assert attention.ndim() == 2
    aeq(attention.shape[0], attention.shape[1], node.shape[0])
    G = build_node_graph(node)
    nnode = node.shape[0]
    G.add_weighted_edges_from([(node.id[i], node.id[j], attention[i, j])
                               for i in range(nnode) for j in range(nnode)])
    return G


def dist_to_long(dist, num=16):
    dist_median = np.median(dist)
    dist[dist > dist_median] = dist_median
    dist = np.ceil(dist / dist_median * (num - 1))
    return dist


def od_to_long(od, num=8):
    do_ = od / od.sum(0)
    od_ = od.T / od.sum(1)
    od_ = np.floor(od_.T / max(od_) * (num - 1))
    do_ = np.floor(do_.T / max(do_) * (num - 1))
    return od_, do_


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
