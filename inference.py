import pandas as pd
import argparse
import joblib

from constants import LABELS
from features import extract_row_features
from models import eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='xgboost', help='Model to use for inference')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--test_path', default='seniority_new.test', help='Path to test data')
    parser.add_argument('--balance_classes', action='store_true', help='Balance classes')
    args = parser.parse_args()

    print('Loading test data...')
    test_df = pd.read_json(args.test_path, lines=True)

    if args.debug:
        test_df = test_df.head(100)

    print('Extracting features and labels...')
    test_features = test_df.apply(extract_row_features, axis=1, result_type='expand')
    labels = pd.Categorical(test_df['seniority_level'], categories=LABELS).codes

    print('Loading model...')
    if args.balance_classes:
        model = joblib.load(f'{args.model}_balanced.joblib')
    else:
        model = joblib.load(f'{args.model}.joblib')

    print('Evaluating on test set...')
    model_name = f'{args.model}_balanced' if args.balance_classes else args.model
    eval(model, model_name, test_features, labels, split='test')

if __name__ == '__main__':
    main()