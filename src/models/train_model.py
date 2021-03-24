from dataset import PtbEcgDataset
import click
from torch.nn.functional import binary_cross_entropy
from torch import square
from torch import sum
import torch.optim as optim
from gnn_models import *
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from visdom import Visdom
import torch


def generation_classification_loss(generated_graph,
                                   original_graph,
                                   classification_predictions,
                                   classification_labels,
                                   h_log_std,
                                   h_mean):
    """Computes the overall loss

    Combines the generator and classification loss to find the overall cost. This cost contains the ELBO loss and the
    classification loss.

    Args:
        generated_graph:
            The graph generated by the generator
        original_graph:
            The original graph fed into the generator
        classification_predictions:
            The classifier's prediction
        classification_labels:
            The labels for the classifier
        h_log_std:
            The log std computed by the generator
        h_mean:
            The mean computed by the generator
    """
    # Compute the reconstruction loss
    reconstruction_loss = binary_cross_entropy(generated_graph, original_graph)

    # Compute the KL loss
    kl_loss = torch.mean(sum(1 + 2*h_log_std - square(h_mean) - square(exp(h_log_std)), dim=1))

    # Compute the classification loss
    classification_loss = binary_cross_entropy(classification_predictions, classification_labels)

    # Add the classification loss
    cost = reconstruction_loss - kl_loss + 100*classification_loss

    return cost


@click.command()
@click.argument('train_data_dir', type=click.Path(exists=True))
@click.argument('train_label_dir', type=click.Path(exists=True))
@click.argument('history_path', type=str)
def train(train_data_dir, train_label_dir, history_path):

    torch.manual_seed(10)

    # Load the dataset
    dataset = PtbEcgDataset(input_data_csv_file=train_data_dir, input_label_csv_file=train_label_dir)

    # Make validation and training split in a stratified fashion
    train_idx, val_idx = train_test_split(np.arange(len(dataset)),
                                          test_size=0.2,
                                          random_state=40,
                                          stratify=dataset.label,
                                          shuffle=True)

    train_dataset = DataLoader(dataset, sampler=SubsetRandomSampler(train_idx))
    val_dataset = DataLoader(dataset, sampler=SubsetRandomSampler(val_idx))

    print('Training dataset has {} samples'.format(len(train_dataset)))
    print('Validation dataset has {} samples'.format(len(val_dataset)))

    # Define classification and generation models
    generator_model = VariationalGraphAutoEncoder(input_dim=6, hidden_dim_1=4, hidden_dim_2=2, num_nodes=15)
    classifier_model = BinaryGraphClassifier(input_dim=6, hidden_dim=4)

    # Optimizers for the classification and generator process
    graph_generator_optimizer = optim.Adam(generator_model.parameters(), lr=1e-4, weight_decay=1e-4)
    graph_classifier_optimizer = optim.Adam(classifier_model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Scheduler
    # scheduler_gen = torch.optim.lr_scheduler.MultiStepLR(graph_classifier_optimizer, milestones=[20, 50, 10], gamma=0.3)
    # scheduler_cl = torch.optim.lr_scheduler.MultiStepLR(graph_classifier_optimizer, milestones=[100, 200, 300], gamma=0.5)

    # Initialize visualizer
    vis = Visdom()

    # Holds the maximum validation accuracy
    max_validation_acc = 0

    for epoch in range(10000):

        # Put models in training mode
        generator_model.train()
        classifier_model.train()

        y_true = list()
        y_pred = list()
        epoch_loss = 0

        for features, label in train_dataset:

            generated_graph = generator_model(torch.ones((15, 15)), features[0])

            classification_predictions = classifier_model(generated_graph.detach(), features[0])

            y_true.append(label.numpy().flatten())
            y_pred.append(classification_predictions.detach().numpy().flatten())

            # ELBO Loss
            loss = generation_classification_loss(generated_graph=generated_graph,
                                                  original_graph=torch.ones((15, 15)),
                                                  classification_predictions=classification_predictions,
                                                  classification_labels=label[0],
                                                  h_log_std=generator_model.h_log_std,
                                                  h_mean=generator_model.h_mean)

            graph_generator_optimizer.zero_grad()
            graph_classifier_optimizer.zero_grad()

            loss.backward()

            graph_generator_optimizer.step()
            graph_classifier_optimizer.step()

            epoch_loss += loss.detach().item()

        epoch_loss /= len(train_dataset)
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        print('Training epoch {}, loss {:.4f}'.format(epoch, epoch_loss))

        # Compute the roc_auc accuracy
        acc = roc_auc_score(y_true.reshape((-1,)), y_pred.reshape(-1,))
        print("Training epoch {}, accuracy {:.4f}".format(epoch, acc))

        vis.line(Y=torch.reshape(torch.tensor(epoch_loss), (-1, )), X=torch.reshape(torch.tensor(epoch), (-1, )),
                 update='append', win='tain_loss',
                 opts=dict(title="Train Losses Per Epoch", xlabel="Epoch", ylabel="Loss"))

        vis.line(Y=torch.reshape(torch.tensor(acc), (-1, )), X=torch.reshape(torch.tensor(epoch), (-1, )),
                 update='append', win='train_acc',
                 opts=dict(title="Train Accuracy Per Epoch", xlabel="Epoch", ylabel="Accuracy"))

        if history_path:
            f = open(history_path+"_train_losses.txt", "a")
            f.write(str(epoch_loss) + "\n")
            f.close()

            f = open(history_path + "_train_accs.txt", "a")
            f.write(str(acc) + "\n")
            f.close()

        with torch.no_grad():
            y_true = list()
            y_pred = list()
            epoch_loss = 0

        generator_model.eval()
        classifier_model.eval()

        for features, label in val_dataset:

            generated_graph = generator_model(adj=torch.ones((15, 15)), features=features[0])

            classification_predictions = classifier_model(generated_graph.detach(), features[0])

            y_true.append(label.numpy().flatten())
            y_pred.append(classification_predictions.detach().numpy().flatten())

            # ELBO Loss
            loss = generation_classification_loss(generated_graph=generated_graph,
                                                  original_graph=torch.ones((15, 15)),
                                                  classification_predictions=classification_predictions,
                                                  classification_labels=label[0],
                                                  h_log_std=generator_model.h_log_std,
                                                  h_mean=generator_model.h_mean)

            epoch_loss += loss.detach().item()

        epoch_loss /= len(val_dataset)
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        print('Validation epoch {}, loss {:.4f}'.format(epoch, epoch_loss))

        # Compute the roc_auc accuracy
        acc = roc_auc_score(y_true.reshape((-1,)), y_pred.reshape(-1,))
        print("Validation epoch {}, accuracy {:.4f}".format(epoch, acc))

        # Save the model only if validation accuracy has increased
        if acc > max_validation_acc:
            print("Accuracy increased. Saving model...")
            torch.save(classifier_model.state_dict(), '../../models/classifier_model.pt')
            torch.save(generator_model.state_dict(), '../../models/generator_model.pt')
            max_validation_acc = acc

        vis.line(Y=torch.reshape(torch.tensor(epoch_loss), (-1,)), X=torch.reshape(torch.tensor(epoch), (-1,)),
                 update='append', win='val_loss',
                 opts=dict(title="Validation Losses Per Epoch", xlabel="Epoch", ylabel="Loss"))

        vis.line(Y=torch.reshape(torch.tensor(acc), (-1,)), X=torch.reshape(torch.tensor(epoch), (-1,)),
                 update='append', win='val_acc',
                 opts=dict(title="Validation Accuracy Per Epoch", xlabel="Epoch", ylabel="Accuracy"))

        if history_path:
            f = open(history_path+"_val_losses.txt", "a")
            f.write(str(epoch_loss) + "\n")
            f.close()

            f = open(history_path + "_val_accs.txt", "a")
            f.write(str(acc) + "\n")
            f.close()

        # Change LR
        # scheduler_gen.step()
        # scheduler_cl.step()


if __name__ == "__main__":
    train()
