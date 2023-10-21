# Seniority Level Predictor

This repository contains a tool for predicting seniority levels using machine learning models. The process involves data splitting, model training, and inference. Below are the instructions and information for using this tool.

## Installation

Create a virtual environment and run the command,

```bash
pip install -r requirements.txt
```

## Usage

### Data Splitting
To generate the training, testing, and validation data splits, execute the following command:

```bash
python split.py
```

## Model Training

To train a model, use the following command, replacing `[MODEL_NAME]` with the name of the desired model. Supported models are:

```bash
python main.py --model [MODEL_NAME] --train_path [TRAIN_PATH] --val_path [VAL_PATH]
```

- `decision_tree`
- `random_forest`
- `xgboost`
- `logistic_regression`
- `neural_network`

You can optionally use the `--balance_classes` flag to balance the classes by increasing the weights of the minority classes.

## Inference

To perform inference on a trained model, run the following command, replacing `[MODEL_NAME]` with the name of the model you want to use. You can also use the `--balance_classes` flag for loading a model trained on balanced class weights.

```bash
python inference.py --model_name [MODEL_NAME]
```

## Features

All feature extraction functions are implemented in the `features.py` file. To add a new feature, create an `extract_feature` function that returns a dictionary with the extracted features. Then, add a line to the `extract_row_features` function to update the main "features" dictionary.

## Models

All machine learning models are implemented in the `models.py` file. To include a new model, add a `get_model_name` function in the `models.py` file and add it to the "MODEL_DICT" for integration into the tool.
