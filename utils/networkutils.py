
import geopandas as gpd
import networkx as nx
import numpy as np

def create_hydro_network(nodes: gpd.GeoDataFrame, links:gpd.GeoDataFrame) -> nx.Graph:
    G = nx.DiGraph()
    G.add_nodes_from(nodes.index.tolist())
    G.add_edges_from([(row.us_node_id, row.ds_node_id) for _, row in links.iterrows()])
    
    for n, p in nodes.iterrows():
        G.nodes[n]['pos'] = (p.x, p.y)

    nx.set_edge_attributes(G,values={(row.us_node_id, row.ds_node_id):name for name, row in links.iterrows()},name="id")
    return G

def get_upstream_nodes(G,node):
    return [n for n in nx.traversal.bfs_tree(G, node, reverse=True) if n != id]

def get_downstream_nodes(G,node):
    return [n for n in nx.traversal.bfs_tree(G, node) if n != id]

def simplify_graph(G:nx.DiGraph)-> nx.DiGraph:
    g = G.copy()
    
    # one in, one out
    degs = (np.array(list(dict(g.in_degree).values())) == 1) & (np.array(list(dict(g.out_degree).values())) == 1)
    g0 = g.copy() #<- simply changing g itself would cause error `dictionary changed size during iteration` 
    for node, deg in zip(g.nodes(),degs):
        if deg:
            if g.is_directed(): #<-for directed graphs
                a0,b0 = list(g.in_edges(node))[0]
                a1,b1 = list(g.out_edges(node))[0]
                
            e0 = a0 if a0!=node else b0
            e1 = a1 if a1!=node else b1

            g0.remove_node(node)
            g0.add_edge(e0, e1)
            g = g0
    return g

def remove_split_edges(g:nx.DiGraph):
    split_nodes = np.array(list(g.nodes))[np.array(list(dict(g.out_degree).values())) == 2]
    for node in split_nodes:
        split_link_1 = [l for l in list(g.edges) if l[0] == node][0]
        g.remove_edge(split_link_1[0],split_link_1[1])
    return


