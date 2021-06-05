import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_test_dataset(filename: str):
    df = pd.read_csv(filename)
    train, test = train_test_split(df, train_size=0.9, random_state=42, stratify=df['label'])

    print(len(train), len(test))
    print(train['label'], test['text'])
    train.to_csv('dataset_training.csv')
    test.to_csv('dataset_test.csv')


def create_balanced_train_dataset(filename: str, posts_to_remove: int):
    df = pd.read_csv(filename)

    indexes = df[df['label'] == 'unrelated'].iloc[posts_to_remove:].index
    df.drop(indexes, inplace=True)
    df.to_csv('dataset_training_balanced.csv')


if __name__ == "__main__":
    create_train_test_dataset(filename='../data/dataset_full.csv')
    create_balanced_train_dataset(filename='../data/dataset_training.csv', posts_to_remove=5771)
