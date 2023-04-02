import pickle

import tqdm

from common.loss import *
from common.utils import *
from model.Graph2Graph import Graph2Graph

if __name__ == '__main__':
    # Data importing
    with open('data/dataset_10_test.pkl', 'rb') as f:
        graphs, target, opt_length = pickle.load(f)

    # Model laoding from file
    model = Graph2Graph(graph_size=10, hidden_dim=200)
    model.load_state_dict("model_G2G_20_2")

    for graph in tqdm.tqdm(dataLoader):
        total_loss = total_examples = prev_loss = 0
        for epoch in range(1, 100000):

            X = torch.concat((batch.x, batch.coordinates), dim=1).to(torch.float32).to(device)
            pi = model.forward(X, batch.edge_index)

            loss = cross_entropy_negative_sampling(pi, batch.y, 5).sum()

            loss.backward()
            optimizer.step()
            total_loss += loss.sum()
            total_examples += pi.numel()
            if epoch % 5 == 0:
                if abs(total_loss - prev_loss) > 10e-6:
                    prev_loss = total_loss
                else:
                    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
                    fig, axs = sample_draw_probs_graph(batch, pi)
                    fig.savefig(f"{PATH}_{plot_counter}.png")
                    plot_counter += 1
                    epoch = 0
                    break
    torch.save(model.state_dict, PATH)
