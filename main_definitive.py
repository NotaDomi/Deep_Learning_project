import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split, ParameterGrid
from steps.visualization import graph_repr, adj_repr

# Visualizzazione dei dispositivi disponibili da usare con PyTorch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Impostazione del seed per la riproducibilità dei risultati
seed = 42
torch.manual_seed(seed)

# Caricamento del grafo dal dataset scaricato
file_path = 'polblogs/polblogs1.gml'
G = nx.read_gml(file_path)

# Visualizzazzione dei primi tre nodi e tre edges per capire come sono strutturati
nodes = list(G.nodes(data=True))
edges = list(G.edges())
print("Nodes:", nodes[:3])
print("Edges:", edges[:3])

# Ottenimento della adjacency matrix e visualizzazione
adj_matrix = nx.adjacency_matrix(G).todense()
adj_repr(adj_matrix)

# Visualizzazione del grafo con colori dei nodi basati sui valori della variabile target
# (blu -> liberali, rosso -> conservatori)
labels = {node: G.nodes[node]['value'] for node in G.nodes()}
graph_repr(G, labels)

# Conversione del grafo di NetworkX in un grafo PyTorch Geometric.
# La matrice di adiacenza in PyTorch Geometric viene rappresentata dal tensore data.edge_index
# in formato COO (Coordinate format) per ottimizzare la computazione
data = from_networkx(G)

# Assegnamento della matrice identità come feature dei nodi (one-hot encoding) non avendo altre feature sui nodi
# Da valutare utilizzo di embedding testuali come features dei nodi
data.x = torch.eye(data.num_nodes).to(device)

# Assegnamento delle etichette (target) ai nodi
data.y = torch.tensor([G.nodes[node]['value'] for node in G.nodes], dtype=torch.long).to(device)

# Divisione dei nodi in train, validation, e test set, utilizzando scikit-learn
train_mask, test_mask = train_test_split(range(data.num_nodes), test_size=0.2, random_state=42)
train_mask, val_mask = train_test_split(train_mask, test_size=0.2, random_state=42)

data.train_mask = torch.tensor([(i in train_mask) for i in range(data.num_nodes)], dtype=torch.bool).to(device)
data.val_mask = torch.tensor([(i in val_mask) for i in range(data.num_nodes)], dtype=torch.bool).to(device)
data.test_mask = torch.tensor([(i in test_mask) for i in range(data.num_nodes)], dtype=torch.bool).to(device)


# Definizione della GCN che verrà utilizzata:
# 3 strati convoluzionali, con una funzione di attivazione relu tra lo strato 1->2, 2->3 e log softmax dopo lo strato 3
# per ottenere la predizione finale. Tra gli strati 1->2 e 2->3 viene anche applicata la regolarizzazione con dropout
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
        self.conv3 = GCNConv(hidden_channels[1], out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


# Definizione del metodo per effettuare il training
def train():
    model.train()
    optimizer.zero_grad()
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    out = model(x, edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


# Definizione del metodo per effettuare la validazione/testing del modello dopo il training
def evaluate(mask):
    model.eval()
    with torch.no_grad():
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        out = model(x, edge_index)
        pred = out[mask].argmax(dim=1)
        correct = (pred == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
    return acc


# Definizione degli iperparametri tra i quali vorremo trovare i valori ottimi
param_grid = {
    'hidden_channels': [[8, 4], [16, 8], [32, 16]],
    'learning_rate': [0.01, 0.001],
    'weight_decay': [5e-4, 5e-5]
}

best_params = None
best_val_acc = 0

# Loop per la valutazione dei diversi modelli e valutazione sul validation set per mantenere i migliori iperparametri
for params in ParameterGrid(param_grid):
    model = GCN(in_channels=data.num_nodes, hidden_channels=params['hidden_channels'], out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    criterion = torch.nn.NLLLoss()

    for epoch in range(200):
        train()

    val_acc = evaluate(data.val_mask)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = params

print(f'Best params: {best_params}, Best Val Accuracy: {best_val_acc:.4f}')

# Addestramento finale con i migliori parametri trovati al passo precendente
model = GCN(in_channels=data.num_nodes, hidden_channels=best_params['hidden_channels'], out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'],
                             weight_decay=best_params['weight_decay'])
criterion = torch.nn.NLLLoss()

for epoch in range(200):
    loss = train()
    if epoch % 10 == 0:
        train_acc = evaluate(data.train_mask)
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}')

# Valutazione dei risultati del modello, allenato con i migliori parametri trovati, utilizzando il test set
test_acc = evaluate(data.test_mask)
print(f'Test Accuracy: {test_acc:.4f}')
