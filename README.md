# Graph Generation Classification Network (GGCN)

## Summary
The Graph Generation Classification Network combines the graph generation and classification processes together. The generation step involves a Variational Graph Autoencoder to create graphs prior to the classification step using Graph Neural Networks. Both the classifier and generator are trained together using a joint objective function. With this approach, graphs can be created for medical datasets without reliance on metadata such as patients' age and gender. The applications of such a network can be summarized as:

* Create graph visualizations of the dataset revealing insight into relationships between samples
* Remove the need for metadata in medical diagnosis
* Augment models using metadata by adding the graph created by the VGAE as another input graph

The overall framework of this network is shown in figure below:

![GGCN](https://github.com/MasoudMo/graph_generation_classification_network/blob/master/docs/GGCN.PNG?raw=true)

## Usage

The dataset can directly be downloaded as a zip file from the following link: 
[PTB Diagnostic Databse](https://www.physionet.org/content/ptbdb/1.0.0/)

Alternatively, the download_data.py script can be used to download the dataset (Please note that the option above is a lot faster.):
```sh
python download_data.py --download_path <path to download dataset to. preferably "../data"> 
```

Before training the model, the raw data has to be processed. The following command can be used to achieve that (Please note that some options are left out purposefully. To see a complete list of available arguments, refer to src/make_dataset.py):
```sh
python make_dataset.py --dataset_dir <path to the raw dataset directory> --output_dir <path to save the processsed data to> --num_comps <number of reduction components>
```

To train the model, the following command can be used (Please note that some options are left out purposefully. To see a complete list of available arguments, refer to src/train.py):
```sh
python train.py --model_dir <path to save the trained models to> --train_data_dir <path to the processed dataset> --train_label_dir <path to dataset labels> --input_dim <input dimension. must match number of reduction components> --epochs <number of training epochs>
```