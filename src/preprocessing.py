# CardGuard - Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Categorical Features

CATEGORICAL_CANDIDATES = [
    "merchant",
    "category",
    "gender",
    "city",
    "state",
    "job",
    "zip",
    "amount_bucket"
]

# Get Feature Types

def get_feature_types(X):
    """
    Split features into categorical and numerical columns.

    The same logic used during model training is used here.
    """

    categorical_features = [
        col
        for col in CATEGORICAL_CANDIDATES
        if col in X.columns
    ]

    numeric_features = [
        col
        for col in X.columns
        if col not in categorical_features
    ]

    return numeric_features, categorical_features

# Create Preprocessor

def create_preprocessor(X):
    """
    Create the CardGuard preprocessing pipeline.

    Numerical:
        Median imputation
        StandardScaler

    Categorical:
        Most-frequent imputation
        OneHotEncoder
    """

    numeric_features, categorical_features = get_feature_types(X)

    numerical_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=True))
    ])

    preprocessor = ColumnTransformer([
        ( "numerical", numerical_pipeline, numeric_features),
        ("categorical",categorical_pipeline,categorical_features)
    ])

    return preprocessor