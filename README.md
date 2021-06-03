# Spatio-Temporal Joint Convolutional Network for Traffic Forecasting

## Structure

* config: the experimetal settings of our model
* data: includes the PEMSD4 and PEMSD8 datasets used in our experiments
* logs: contains the logs during training or testing
* model: save the model (we also provide pre-trained models for both datasets)
* model.py: implement of our STJGCN model
* utils.py: tools, including data processing, evaluation metrics, etc.
* tf_utils.py: tensorflow-based tools
* train.py: code of training STJGCN
* test.py: code of testing STJGCN

## Requirements

Python 3.7.10, tensorflow 1.14.0, numpy 1.20.3, scipy 1.2.1, argparse and configparser

## Training

To train STJGCN on the PeMSD4 or PeMSD8 dataset, run this command:

```train
python train.py --config config/STJGCN_PeMSD4.conf
python train.py --config config/STJGCN_PeMSD8.conf
```

## Evaluation

To evaluate STJGCN on the PeMSD4 or PeMSD8 dataset, run:

```eval
python test.py --config config/STJGCN_PeMSD4.conf
python test.py --config config/STJGCN_PeMSD8.conf
```

## Results

We provide pre-trained models on both datasets, which achieve the following performance:

| Dataset |  MAE  |  RMSE  |  MAPE  |
| --------|------ | ------ | ------ |
| PeMSD4  | 18.83 | 30.32  | 11.89% |
| PeMSD8  | 14.51 | 23.82  |  9.06% |

Note that this result is different from (better than) Table 1 in the paper, because we report the average error over 10 runs in Table 1.
