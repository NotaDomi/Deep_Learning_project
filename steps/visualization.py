import networkx as nx
from matplotlib import pyplot as plt


# Definizione del metodo che permette di rappresentare il grafo. Accetta come parametri il grafo networkx
# e le etichette dei nodi. Grazie alle etichette possiamo colorare i nodi in base alla classe di appartenenza
def graph_repr(g, labels):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(g)
    node_colors = ['red' if labels[node] == 1 else 'blue' for node in g.nodes()]
    nx.draw(g, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=500, font_size=10)
    plt.title('Grafo dei Blog con Hyperlink')
    plt.show()


# Definizione del metodo che permette di rappresentare la matrice di adiacenza.
# Accetta come parametro la matrice di adiacenza ottenuta da networkx
def adj_repr(adj):
    plt.figure(figsize=(15, 15))
    plt.imshow(adj, cmap='Blues', interpolation='none')
    plt.title('Matrice di Adiacenza del Grafo')
    plt.xlabel('Nodi')
    plt.ylabel('Nodi')
    plt.show()
