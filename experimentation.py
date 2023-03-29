import pickle

import torch
import tqdm
from torch_geometric.loader import DataLoader

from common.loss import cross_entropy_negative_sampling
from common.utils import sample_draw_probs_graph
from model.Graph2Graph import Graph2Graph

PATH = "model_G2G_20"


if __name__ == '__main__':
    # Data importing
    with open('data/dataset_20_train.pkl', 'rb') as f:
        graphs, target, opt_length = pickle.load(f)
        dataLoader = DataLoader(graphs, batch_size=128)

    # Model Initialization
    model = Graph2Graph(graph_size=20, hidden_dim=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    plot_counter = 0
    for epoch in range(1, 500):
        total_loss = total_examples = prev_loss = 0
        for batch in tqdm.tqdm(dataLoader):
            optimizer.zero_grad()

            X = batch.x.to(torch.float32)
            edge_attr = batch.edge_attr.to(torch.float32)
            edge_index = batch.edge_index

            pi = model.forward(X, edge_index, edge_attr)

            loss = cross_entropy_negative_sampling(pi, batch.y, 10).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += loss.sum()
            total_examples += pi.numel()
        if abs(total_loss - prev_loss) > 10e-6:
            prev_loss = total_loss
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
        fig, axs = sample_draw_probs_graph(batch, pi)
        fig.savefig(f"{PATH}_GAT_{plot_counter}.png")
        plot_counter += 1
        torch.save(model.state_dict, PATH)
