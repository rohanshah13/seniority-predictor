import pandas as pd
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from constants import LABELS
from features import extract_row_features
from models import train, eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='xgboost', help='Model to train')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--train_path', default='seniority_new.train', help='Path to train data')
    parser.add_argument('--val_path', default='seniority_new.val', help='Path to val data')
    parser.add_argument('--balance_classes', action='store_true', help='Balance classes')
    args = parser.parse_args()

    print('Reading data...')
    train_df = pd.read_json(args.train_path, lines=True)
    val_df = pd.read_json(args.val_path, lines=True)

    if args.debug:
        train_df = train_df.head(100)
        val_df = val_df.head(100)
    
    # Drop the rows with missing labels and print the number of rows dropped
    train_df = train_df.dropna(subset=['seniority_level'])
    val_df = val_df.dropna(subset=['seniority_level'])

    print('Extracting features and labels...')
    train_features = train_df.apply(extract_row_features, axis=1, result_type='expand')
    train_labels = pd.Categorical(train_df['seniority_level'], categories=LABELS).codes
    val_features = val_df.apply(extract_row_features, axis=1, result_type='expand')
    val_labels = pd.Categorical(val_df['seniority_level'], categories=LABELS).codes

    print('Training models...')
    model = train(args.model, train_features, train_labels, balance_classes=args.balance_classes)

    print('Evaluating on val set...')
    model_name = f'{args.model}_balanced' if args.balance_classes else args.model

    eval(model, model_name, val_features, val_labels, split='val')


if __name__ == '__main__':
    main()