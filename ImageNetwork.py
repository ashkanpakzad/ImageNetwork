import networkx as nx
import numpy as np

def connections(conn_matrix):
    '''Internal function to ensure connectivty matrix is valid and get indices of connectivity relative to centre.'''
    if np.any(conn_matrix != conn_matrix.T):
        raise ValueError('Connectivity matrix must be symmetric.')
    conn_idx = np.argwhere(conn_matrix)
    conn_idx = conn_idx - np.array(conn_matrix.shape)//2
    return conn_idx

def neighbor(idx, conn_idx):
    '''Internal function to get neighbor indices of given index.'''
    nb_ind = idx + conn_idx
    return nb_ind

def ImageNetwork(binaryimage, conn):
    '''Create an image network for a given N-dimensional binary image with specified connectivity
    (see https://en.wikipedia.org/wiki/Pixel_connectivity).

    e.g.
    # set up binary image
    binaryimage = np.zeros((16, 16), dtype=int)
    binaryimage[16 // 2 - 16 // 4:16 // 2 + 16 // 4, 16 // 2 - 16 // 4:16 // 2 + 16 // 4] = 1
    # for eight pixel connectivity
    conn = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    G, nodeind = ImageNetwork.ImageNetwork(binaryimage, conn)'''
    # set up connectivity indices
    idx = connections(conn)
    idx = idx[0:idx.shape[0] // 2] # discard half of the connections un-needed duplicates.

    # set up graph and identify each nonzero voxel as a node.
    G = nx.Graph()
    nodeind = np.argwhere(binaryimage)
    nodelist = np.arange(len(nodeind))+1
    G.add_nodes_from(nodelist)

    # create a node image where intensities = node number
    nodeimage = np.zeros_like(binaryimage)
    nodeimage[tuple(map(tuple, nodeind.T))] = nodelist

    for i in nodelist:
        nb = neighbor(nodeind[i-1], idx) # identify neighbor indices
        nb_node = nodeimage[tuple(map(tuple, nb.T))] # identify neighbor nodes
        nb_edges = [(i, node) for node in nb_node] # generate edge list
        G.add_edges_from(nb_edges) # add edges to neighbor nodes.

    G.remove_node(0) # connections made to node '0' that represents pixels outside of binary object.
    return G, nodeind

def setweight(G, nodeind, stepcost):
    '''Set weights of the image graph G after its creation using ImageNetwork. nodeind returned by Image network. Create
    a function 'stepcost' for how the weights should be set between nodes such that input is source and terminal nodes
    of edges and output is the weight of that edge.

    e.g
    # euclidean distance weights
    def stepcost(s, t):
        return np.sqrt((s[1] - t[1]) ** 2 + (s[0] - t[0]) ** 2)'''
    for u, v in G.edges:
        # get index of edges for nodeind list
        snode = u - 1
        tnode = v - 1
        # get node pixel index
        s = nodeind[snode]
        t = nodeind[tnode]
        # compute weight of edge based on step cost function provided
        G[u][v]['weight'] = stepcost(s, t)
    return G