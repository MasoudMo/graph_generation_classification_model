from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class PtbEcgDataset(Dataset):
    """The PTB ECG dataset

    The dataset wrapper to accommodate the training process using the PTB ECG dataset

    """

    def __init__(self, input_data_csv_file, input_label_csv_file):
        """Loads the dataset's CSV file

        Loads the input data CSV file to be used with the dataloader during training

        Args:
            input_data_csv_file:
                The csv file containing the ECG data
            input_label_csv_file:
                The csv file containing each sample's label
        """

        pd_df = pd.read_csv(input_data_csv_file)
        self.data = pd_df.to_numpy(dtype=np.float64)

        pd_df = pd.read_csv(input_label_csv_file)
        self.label = pd_df.to_numpy(dtype=np.int8)

    def __getitem__(self, item):
        """Return the 15 channels of ECG data

        Returns the 15 ECG data channels for the input index item

        Args:
            item:
                The iterator for which the data is retrieved

        Returns:
            the 15-channel data corresponding the the input index item and its label

        """

        sample_data = self.data[item*15: (item+1)*15, :]
        sample_label = self.label[item]

        return sample_data, sample_label

    def __len__(self):
        """Returns the dataset size

        Returns:
            The size of the dataset

        """

        return self.data.shape[0]
