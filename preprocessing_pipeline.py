import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List


def load_and_combine_data(file_paths: List[str]) -> pd.DataFrame:
    """Loads data from multiple CSV files and concatenates them."""
    dfs = [pd.read_csv(path) for path in file_paths]
    return pd.concat(dfs, ignore_index=True)


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing values using the mode for specified columns."""
    # Using median is more robust for skewed numerical data like this
    for col in ['1h', '24h', '7d', '24h_volume']:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies log transformation to skewed numerical features."""
    for col in ['price', '24h_volume', 'mkt_cap']:
        df[col] = np.log1p(df[col])
    return df


def create_preprocessing_pipeline(
    file_paths: List[str],
    top_n_coins: int = 10,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, LabelEncoder, StandardScaler]:
    """
    Full preprocessing pipeline for the cryptocurrency liquidity data.

    Args:
        file_paths (List[str]): List of paths to the input CSV files.
        top_n_coins (int): The number of top coins to consider individually.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.

    Returns:
        A tuple containing:
        - X_train_scaled: Scaled training features.
        - X_test_scaled: Scaled testing features.
        - y_train: Training target variable.
        - y_test: Testing target variable.
        - label_encoder: Fitted LabelEncoder for the 'coin' feature.
        - scaler: Fitted StandardScaler for the numerical features.
    """
    df = load_and_combine_data(file_paths)

    # --- 1. Imputation & Initial Cleaning ---
    df = impute_missing_values(df)

    # --- 2. Feature Engineering: Add Time-Based Features ---
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df.drop(columns=['date', 'symbol'], inplace=True, errors='ignore')

    # --- 3. Log Transformation ---
    df = transform_features(df)

    # --- 4. Train-Test Split (Crucial to do this BEFORE fitting encoders/scalers) ---
    X = df.drop('24h_volume', axis=1)
    y = df['24h_volume']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --- 5. Handle High-Cardinality 'coin' Feature (Leakage-Free) ---
    # Determine top coins based on mean volume *only from the training set*
    train_volume = X_train.join(y_train).groupby('coin')['24h_volume'].mean().reset_index()
    top_coins = train_volume.sort_values(by='24h_volume', ascending=False).head(top_n_coins)['coin'].tolist()
    top_coins_with_other = top_coins + ['Other']

    # Apply grouping to both train and test sets
    X_train['coin_grouped'] = X_train['coin'].apply(lambda x: x if x in top_coins else 'Other')
    X_test['coin_grouped'] = X_test['coin'].apply(lambda x: x if x in top_coins else 'Other')

    # Fit LabelEncoder *only on the defined training categories*
    label_encoder = LabelEncoder().fit(top_coins_with_other)
    X_train['coin_grouped_encoded'] = label_encoder.transform(X_train['coin_grouped'])
    X_test['coin_grouped_encoded'] = label_encoder.transform(X_test['coin_grouped'])

    X_train.drop(columns=['coin', 'coin_grouped'], inplace=True)
    X_test.drop(columns=['coin', 'coin_grouped'], inplace=True)

    # --- 6. Scaling Numerical Features ---
    features_to_scale = [col for col in X_train.columns if col != 'coin_grouped_encoded']
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
    X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])

    # --- 7. Optimize Categorical Feature for LightGBM ---
    X_train_scaled['coin_grouped_encoded'] = X_train_scaled['coin_grouped_encoded'].astype('category')
    X_test_scaled['coin_grouped_encoded'] = X_test_scaled['coin_grouped_encoded'].astype('category')

    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler


def preprocess():
    file1 = 'data/coin_gecko_2022-03-16.csv'
    file2 = 'data/coin_gecko_2022-03-17.csv'

    X_train_scaled, X_test_scaled, y_train, y_test, le, scaler = create_preprocessing_pipeline([file1, file2])

    print("--- X_train_scaled Head ---")
    print(X_train_scaled.head())
    print(f"\nTraining data shape: {X_train_scaled.shape}")
    print(f"Testing data shape: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, le, scaler