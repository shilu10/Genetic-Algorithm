from sklearn.model_selection import train_test_split 
import imblearn
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import LabelEncoder

def preprocess_dataframe(df): 
    """
        This Function, will be used to drop unwanted cols, and it converts the categorical data into numerical.
        Params:
            df: dataframe object that needed to be processed
    """
    df.drop(["customerID"], inplace=True, axis=1)
    le = LabelEncoder()
    le.fit(df.Churn)
    churn = le.transform(df.Churn)
    df.Churn = churn
    
    df[df.TotalCharges == " "] = 0
    df.TotalCharges = df.TotalCharges.apply(lambda x: float(x))
    df.MonthlyCharges = df.MonthlyCharges.astype("float")
    df = pd.get_dummies(df)
    return df

def handle_imbalance(train_X, train_y): 
    """
        This function, does handle a imbalance data, by doing the oversampling using ADASYN method from imblearn.
        Params:
            train_X: training x, that needed to be resampled.
            train_y: training y, that is of train x
    """
    ada = ADASYN(random_state=42)
    train_X, train_y = ada.fit_resample(train_X, train_y)
    return train_X, train_y

def get_training_testing_data(dataframe, out_col, train_size): 
    """
        Params:
            dataframe: datafram, that needed to be processed.
            out_col: output column for the y(label).
            test_split: split size.
    """
    y = dataframe[out_col]
    dataframe.drop([out_col], inplace=True, axis=1)
    X = dataframe
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=train_size, stratify=y)
    train_y = train_y.astype("int")
    test_y = test_y.astype("int")

    return train_X, train_y, test_X, test_y