from dataset import PtbEcgDataset
import click
from torch.nn import BCELoss
import torch.optim as optim
from gnn_models import *
import torch


@click.command()
@click.argument('train_data_dir', type=click.Path(exists=True))
@click.argument('train_label_dir', type=click.Path(exists=True))
def train(train_data_dir, train_label_dir):

    dataset = PtbEcgDataset(input_data_csv_file=train_data_dir, input_label_csv_file=train_label_dir)

    print('Training dataset has {} samples'.format(len(dataset)/15))

    generator_model = VariationalGraphAutoEncoder(input_dim=14, hidden_dim_1=10, hidden_dim_2=5, num_nodes=15)
    classifier_model = BinaryGraphClassifier(input_dim=5, hidden_dim=5)

    loss_func = BCELoss()
    graph_generator_optimizer = optim.Adam(generator_model.parameters(), lr=0.01)
    graph_classifier_optimizer = optim.Adam(classifier_model.parameters(), lr=0.01)

    for epoch in range(100):

        generator_model.train()
        classifier_model.train()
        print(generator_model)
        for graph, label in dataset:

            generated_graph = generator_model(graph)
            loss = loss_func(generated_graph, torch.ones((15, 15)))

            graph_generator_optimizer.zero_grad()

            loss.backward()

            graph_generator_optimizer.step()

            print(loss.item())

if __name__ == "__main__":
    train()
