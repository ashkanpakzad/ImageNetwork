import networkx as nx
import numpy as np
import ImageNetwork

import matplotlib.pyplot as plt

# set up binary image
imsz = 16
binaryimage = np.zeros((imsz, imsz), dtype=int)

binaryimage[imsz // 2 - imsz // 4:imsz // 2 + imsz // 4, imsz // 2 - imsz // 4:imsz // 2 + imsz // 4] = 1

# set up desired connectivity
conn = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) # for eight connectivity

# convert to graph
G, nodeind = ImageNetwork.ImageNetwork(binaryimage, conn)

# set weights according to euclidean distance
def stepcost(s, t):
    return np.sqrt((s[1] - t[1]) ** 2 + (s[0] - t[0]) ** 2)

G = ImageNetwork.setweight(G, nodeind, stepcost)

# save graph for output
nx.write_gexf(G, "ImageNetwork_test.gexf")
# plot the graph
plt.figure()
nx.draw_kamada_kawai(G, with_labels=True)
plt.figure()
plt.imshow(binaryimage)
plt.show()
