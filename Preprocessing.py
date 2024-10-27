import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

class Preprocessing:
    def __init__(self, df, label_col, training=True, num_imputer=None, cat_imputer=None, one_hot_encoder=None):
        # Drop 'id' column and any columns starting with 'PCIAT-'
        df = df.drop(columns=['id'], errors='ignore')
        pciat_columns = [col for col in df.columns if col.startswith('PCIAT-')]
        df = df.drop(columns=pciat_columns, errors='ignore')

        # Drop specific season-related columns
        # TODO: Discuss with team about missing columns

        other_columns = [
            'Fitness_Endurance-Season',
            'PAQ_A-Season',
            'BIA-Season',
            'PAQ_A-Season'
        ]

        df = df.drop(columns=other_columns, errors='ignore')

        # Remove rows with missing labels for training data
        if training:
            df = df.dropna(subset=[label_col])
        
        # Set attributes
        self.df = df
        self.label_col = label_col
        self.training = training
        self.features = self.df.drop(columns=[label_col]) if label_col in self.df else self.df
        self.label = self.df[label_col].clip(lower=0, upper=3) if label_col in self.df else None
        
        # Identify numeric and categorical columns
        self.numeric_columns = self.features.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.features.select_dtypes(include=[object]).columns

        # Initialize imputers and encoders
        self.num_imputer = num_imputer
        self.cat_imputer = cat_imputer
        self.one_hot_encoder = one_hot_encoder

        # Automatically run preprocessing
        self._preprocess()

    def _preprocess(self):
        """Automatically handle imputation and encoding based on training or test mode."""
        # Impute data
        self._impute_data()

        # One-hot encode features
        self._one_hot_encode_features()

        # One-hot encode labels if available
        if self.label is not None:
            self._one_hot_encode_labels()

    def _impute_data(self):
        """Apply simple imputation to both numeric (mean) and categorical (most frequent) features."""
        if self.training:
            # Fit imputers on training data and store them
            self.num_imputer = SimpleImputer(strategy='mean')
            self.features[self.numeric_columns] = self.num_imputer.fit_transform(self.features[self.numeric_columns])

            self.cat_imputer = SimpleImputer(strategy='most_frequent')
            self.features[self.categorical_columns] = self.cat_imputer.fit_transform(self.features[self.categorical_columns])
        else:
            # Use stored imputers for test data
            self.features[self.numeric_columns] = self.num_imputer.transform(self.features[self.numeric_columns])
            self.features[self.categorical_columns] = self.cat_imputer.transform(self.features[self.categorical_columns])

    def _one_hot_encode_features(self):
        """Convert categorical features to one-hot encoded representation."""
        ohe_columns = pd.get_dummies(self.features[self.categorical_columns], drop_first=False, dtype=float)
        self.features = pd.concat([self.features.drop(columns=self.categorical_columns), ohe_columns], axis=1)

    def _one_hot_encode_labels(self):
        """Convert labels to one-hot encoded vectors."""
        self.label.fillna(0, inplace=True)
        if self.training:
            self.one_hot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
            label_encoded = self.one_hot_encoder.fit_transform(self.label.values.reshape(-1, 1))
        else:
            label_encoded = self.one_hot_encoder.transform(self.label.values.reshape(-1, 1))
        
        # Update label with one-hot encoded values
        self.label = pd.DataFrame(label_encoded, columns=[f"sii_{int(i)}" for i in range(4)])
