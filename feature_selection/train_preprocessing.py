from sklearn.model_selection import train_test_split 

def get_training_testing_data(dataframe, out_col, train_split): 
    """
        Params:
            dataframe: datafram, that needed to be processed.
            out_col: output column for the y(label).
            test_split: split size.
    """
    y = dataframe[out_col]
    dataframe.drop([out_col], inplace=True, axis=1)
    X = dataframe
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_split=train_split)

    return train_X, train_y, test_X, test_y