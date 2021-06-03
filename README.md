# Spatio-Temporal Joint Convolutional Network for Traffic Forecasting

This repository is the official implementation of [Spatio-Temporal Joint Convolutional Network for Traffic Forecasting] 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --config config/STJGCN_PeMSD4.conf
python train.py --config config/STJGCN_PeMSD8.conf
```

## Evaluation

To evaluate my model on the PeMSD4 dataset, run:

```eval
python test.py --config config/STJGCN_PeMSD4.conf
python test.py --config config/STJGCN_PeMSD8.conf
```

## Results

Our model achieves the following performance:

| Dataset |  MAE  |  RMSE  |  MAPE  |
| --------|------ | ------ | ------ |
| PeMSD4  | 18.83 | 30.32  | 11.89% |
| --------|------ | ------ | ------ |
| PeMSD8  | 14.51 | 23.82  | 9.06%  |
