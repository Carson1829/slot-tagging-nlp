import pandas as pd
import torch
from data import load_data, tokenize, build_vocab, build_tag_vocab


def main():


    # Reading Train Dataset into DataFrame
    train_data = pd.read_csv('data/train.csv')

    # Splitting the Training dataset into the Training set and Validation set
    # train_data, val_data = train_test_split(df, random_state = 32, test_size = 0.25, shuffle = True)
    print('Train Data Shape: ', train_data.shape)
    # print('Val Data Shape: ', val_data.shape)
    test_data = pd.read_csv('data/test.csv')
    print('Test Set Shape:', test_data.shape)

    train_sents, train_tags = load_data(train_data, has_tags=True)
    test_sents = load_data(test_data)
    print(f"{train_sents[0]}")



if __name__ == "__main__":
    main()
