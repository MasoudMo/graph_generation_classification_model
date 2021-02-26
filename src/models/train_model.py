from dataset import PtbEcgDataset
import click


@click.command()
@click.argument('train_data_dir', type=click.Path(exists=True))
@click.argument('train_label_dir', type=click.Path(exists=True))
def train(train_data_dir, train_label_dir):

    dataset = PtbEcgDataset(input_data_csv_file=train_data_dir, input_label_csv_file=train_label_dir)

    print('Training dataset has {} samples'.format(len(dataset)/15))

    for data, label in dataset:
        print(data.shape)
        print(label)


if __name__ == "__main__":
    train()
