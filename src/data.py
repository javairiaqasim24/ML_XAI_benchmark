import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PATH, ID_COLUMN, RANDOM_STATE, TARGET, TEST_SIZE


def load_and_clean_data(path=DATA_PATH):
    """Load the Telco dataset and reproduce the notebook cleaning steps."""
    df = pd.read_csv(path)

    df_clean = df.copy()
    df_clean = df_clean.drop(ID_COLUMN, axis=1)

    # The original dataset stores TotalCharges as text.
    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean["TotalCharges"], errors="coerce"
    )

    # The notebook replaces the resulting missing TotalCharges with 0.
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0)

    df_clean[TARGET] = df_clean[TARGET].map({"No": 0, "Yes": 1})

    return df_clean


def make_train_test(df_clean):
    """Create the same stratified 80/20 split used in the notebook."""
    X = df_clean.drop(TARGET, axis=1)
    y = df_clean[TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
