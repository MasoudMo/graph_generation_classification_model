# -*- coding: utf-8 -*-
import logging
import os
import wfdb
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse


def main():

    # Command line argument parser
    parser = argparse.ArgumentParser(description='Data preprocessing for GGCN')
    parser.add_argument('--dataset_dir',
                        type=str,
                        required=False,
                        default="../data/",
                        help='Path to the raw dataset.')
    parser.add_argument('--output_dir',
                        type=str,
                        required=False,
                        default="../data/processed_data/",
                        help='Path to save the processed data to.')
    parser.add_argument('--save_raw_data',
                        type=bool,
                        required=False,
                        default=None,
                        help='Indicates whether raw data is saved or not.')
    parser.add_argument('--save_filtered_data',
                        type=bool,
                        required=False,
                        default=None,
                        help='Indicates whether filtered data is saved or not.')
    parser.add_argument('--ecg_samp_to',
                        type=int,
                        required=False,
                        default=20000,
                        help='Indicates the last sample index used from the ECG signal.')
    parser.add_argument('--num_comps',
                        type=int,
                        required=False,
                        default=10,
                        help='Indicates the number of reduction components to use.')
    parser.add_argument('--cutoff_freq',
                        type=int,
                        required=False,
                        default=10,
                        help='Indicates the cutoff frequency of the high pass filter.')
    parser.add_argument('--ctrl_repeats',
                        type=int,
                        required=False,
                        default=4,
                        help='Indicates the number of times healthy samples are repeated for augmentation.')
    parser.add_argument('--reduction_type',
                        type=str,
                        required=False,
                        default='tsne',
                        help='Indicates the reduction type (pca or tsne).')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    save_raw_data = args.save_raw_data
    save_filtered_data = args.save_filtered_data
    ecg_samp_to = args.ecg_samp_to
    num_comps = args.num_comps
    cutoff_freq = args.cutoff_freq
    ctrl_repeats = args.ctrl_repeats
    reduction_type = args.reduction_type

    logger = logging.getLogger(__name__)
    logger.info('Processing raw data...')

    # Some records have missing clinical summary
    records_to_exclude = [421, 348, 536, 338, 358, 429, 395, 377, 419, 398, 367, 412, 416, 522, 333, 523, 378,
                          375, 397, 519, 530, 406, 524, 355, 356, 407, 417]

    # The dataset comes with a records file including path to each patient's data
    record_files = open(os.path.join(dataset_dir, 'RECORDS'))
    record_files = [os.path.join(dataset_dir, file) for file in record_files.read().split('\n')][:-1]
    record_files = record_files[:-1]
    record_files = [i for j, i in enumerate(record_files) if j not in records_to_exclude]

    # Diagnosis dictionary
    diagnosis = {
        'Reason for admission: Healthy control': 0,
        'Reason for admission: Myocardial infarction': 1,
        'Reason for admission: Heart failure (NYHA 2)': 2,
        'Reason for admission: Bundle branch block': 3,
        'Reason for admission: Dysrhythmia': 4,
        'Reason for admission: Myocardial hypertrophy': 5,
        'Reason for admission: Valvular heart disease': 6,
        'Reason for admission: Myocarditis': 7,
        'Reason for admission: Hypertrophy': 8,
        'Reason for admission: Cardiomyopathy': 9,
        'Reason for admission: Heart failure (NYHA 3)': 10,
        'Reason for admission: Unstable angina': 11,
        'Reason for admission: Stable angina': 12,
        'Reason for admission: Heart failure (NYHA 4)': 13,
        'Reason for admission: Palpitation': 14}

    # Extract the number of samples in the dataset
    logger.info('Number of original data samples: {}'.format(len(record_files)))

    # Variable holding record labels
    labels = []

    # Compute number of records after processing (including augmentation)
    num_processed_records = 80*ctrl_repeats+442
    logger.info('Number of processed data samples: {}'.format(num_processed_records))

    # Variable holding all filtered data (to be used for PCA)
    all_filtered_data = np.empty((num_processed_records*15, ecg_samp_to))
    reduced_filtered_data = np.empty((num_processed_records*15, num_comps))

    # Extract each record and save it to the CSV file
    curr_idx = 0
    for i, record in enumerate(record_files):

        # Extract record data
        record_data = wfdb.io.rdrecord(record, sampto=ecg_samp_to)

        # Extract the label
        labels.append(diagnosis[record_data.comments[4]])

        # Read the raw channel data
        raw_data = np.transpose(record_data.p_signal)

        # Save the raw data
        if save_raw_data:
            np.savetxt(os.path.join(output_dir, "raw_data_"+str(curr_idx)+".csv"), raw_data, delimiter=',')

        # Filter the data (high pass filter)
        # noinspection PyTupleAssignmentBalance
        b, a = signal.butter(N=5, Wn=cutoff_freq / (record_data.fs/2), btype='highpass', output='ba')
        filtered_data = np.empty_like(raw_data)
        for idx in range(raw_data.shape[0]):
            filtered_data[idx, :] = signal.filtfilt(b, a, raw_data[idx, :])

        # Save the filtered data
        if save_filtered_data:
            np.savetxt(os.path.join(output_dir, "filtered_data_"+str(curr_idx)+".csv"), filtered_data, delimiter=',')

        all_filtered_data[curr_idx*15: (curr_idx+1)*15, :] = filtered_data
        curr_idx += 1

        if labels[-1] is 0:
            for j in range(1, ctrl_repeats):

                # Extract record data
                record_data = wfdb.io.rdrecord(record, sampfrom=j*ecg_samp_to, sampto=(j+1)*ecg_samp_to)

                # Extract the label
                labels.append(0)

                # Read the raw channel data
                raw_data = np.transpose(record_data.p_signal)

                # Save the raw data
                if save_raw_data:
                    np.savetxt(os.path.join(output_dir, "raw_data_"+str(curr_idx)+".csv"), raw_data, delimiter=',')

                # Filter the data (high pass filter)
                # noinspection PyTupleAssignmentBalance
                b, a = signal.butter(N=5, Wn=cutoff_freq / (record_data.fs/2), btype='highpass', output='ba')
                filtered_data = np.empty_like(raw_data)
                for idx in range(raw_data.shape[0]):
                    filtered_data[idx, :] = signal.filtfilt(b, a, raw_data[idx, :])

                # Save the filtered data
                if save_filtered_data:
                    np.savetxt(os.path.join(output_dir, "filtered_data_"+str(curr_idx)+".csv"), filtered_data, delimiter=',')

                all_filtered_data[curr_idx*15: (curr_idx+1)*15, :] = filtered_data
                curr_idx += 1

    # Perform dimension reduction for each channel
    test_var = all_filtered_data[0::15]
    if reduction_type is 'pca':
        for i in range(15):
            pca = PCA(n_components=num_comps)
            pca.fit(all_filtered_data[0+i::15])
            reduced_filtered_data[0+i::15] = pca.transform(all_filtered_data[0+i::15])
    elif reduction_type is 'tsne':
        reduced_filtered_data = TSNE(n_components=num_comps, method='exact').fit_transform(all_filtered_data)

    # Save the reduced filtered data
    np.savetxt(os.path.join(
        output_dir,
        "reduced_filtered_data.csv"),
        reduced_filtered_data,
        delimiter=','
    )

    # Save labels file
    np.savetxt(os.path.join(output_dir, "labels.csv"), np.array(labels, dtype=int), delimiter=',')

    logger.info('Finished processing data!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
