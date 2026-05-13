import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config

def balance_classes(df):
    """
    Balances the fire_flag classes by undersampling the majority class (flag=0).
    Uses a fixed random_state for reproducible undersampling.
    """
    class_counts = df['fire_flag'].value_counts()
    min_class_size = class_counts.min()
    
    class_0 = df[df['fire_flag'] == 0]
    class_1 = df[df['fire_flag'] == 1]
    
    class_0_balanced = class_0.sample(n=min_class_size, random_state=config.RANDOM_STATE)
    balanced_df = pd.concat([class_0_balanced, class_1], ignore_index=True)
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
    return balanced_df

def prepare_train_test_data(df):
    """
    Splits the balanced dataframe into Train/Test splits,
    and applies a StandardScaler on the feature matrix.
    Returns: x_train, x_test, y_train, y_test
    """
    X = df.drop(columns=['fire_flag'])
    y = df['fire_flag']
    
    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    # Scale variables
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)
    
    return x_train_scaled, x_test_scaled, y_train, y_test

def get_preprocessed_features(df):
    """
    Aggregates balancing and train test splitting functions.
    """
    balanced = balance_classes(df)
    return prepare_train_test_data(balanced)
