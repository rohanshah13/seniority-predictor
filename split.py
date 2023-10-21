import pandas as pd
import argparse

from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='seniority.train', help='Path to train data')
    parser.add_argument('--test_path', default='seniority.test', help='Path to test data')
    args = parser.parse_args()

    df_train = pd.read_json(args.train_path, lines=True)
    df_test = pd.read_json(args.test_path, lines=True)

    # Drop columns without labels
    df_train = df_train.dropna(subset=['seniority_level'])
    df_test = df_test.dropna(subset=['seniority_level'])

    # Split test set into val and test sets
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42)

    # Save the dataframes
    df_train.to_json('seniority_new.train', orient='records', lines=True)
    df_val.to_json('seniority_new.val', orient='records', lines=True)
    df_test.to_json('seniority_new.test', orient='records', lines=True)



if __name__ == '__main__':
    main()