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
        """Automatically handle imputation, outlier capping, encoding, and feature engineering."""
        # Handle missing data and outliers
        self._handle_missing_data()
        self._cap_outliers()

        # Feature engineering
        self._create_derived_features()
        self._simplify_seasonal_features()

        # Re-identify categorical columns after seasonal feature simplification
        self.categorical_columns = self.features.select_dtypes(include=[object]).columns

        # One-hot encode features
        self._one_hot_encode_features()

        # One-hot encode labels if available
        if self.label is not None:
            self._one_hot_encode_labels()

    def _handle_missing_data(self):
        # Applied refined imputation for numeric and categorical features
        if 'Basic_Demos-Age' in self.numeric_columns:
            age_imputer = SimpleImputer(strategy='median') # use of median over mean just for age = Basic_Demos-Age
            self.features['Basic_Demos-Age'] = age_imputer.fit_transform(
                self.features[['Basic_Demos-Age']]
            )
            self.numeric_columns = self.numeric_columns.drop('Basic_Demos-Age')

        if self.training:
            self.num_imputer = SimpleImputer(strategy='mean')
            self.features[self.numeric_columns] = self.num_imputer.fit_transform(self.features[self.numeric_columns])

            self.cat_imputer = SimpleImputer(strategy='most_frequent')  # categorical columns imputed using the most frequent value
            self.features[self.categorical_columns] = self.cat_imputer.fit_transform(
                self.features[self.categorical_columns])
        else:
            self.features[self.numeric_columns] = self.num_imputer.transform(self.features[self.numeric_columns])
            self.features[self.categorical_columns] = self.cat_imputer.transform(
                self.features[self.categorical_columns])

    def _cap_outliers(self):
        # Physical-Height, Physical-Weight, and Physical-BMI set within reasonable limits. Based on RN info.
        # This should help training improving the model stability and robustness. Consult with the team
        outlier_caps = {
            'Physical-Height': (50, 80),
            'Physical-Weight': (30, 120),
            'Physical-BMI': (10, 40)
        }
        for col, (min_val, max_val) in outlier_caps.items():
            if col in self.features.columns:
                self.features[col] = np.clip(self.features[col], min_val, max_val)

    # Feature Engineering (Marcelo)
    def _create_derived_features(self):
        """Create summary statistics or derived features for broader trends."""
        # I believe that summing selected screen time columns might better represent the overall screen exposure
        # consult with the team
        screen_time_columns = ['PreInt_EduHx-computerinternet_hoursday']
        if all(col in self.features.columns for col in screen_time_columns):
            self.features['Total_Screen_Time'] = self.features[screen_time_columns].sum(axis=1)

        # Added Physical_Health_Summary feature and calculate the mean of physical health metrics: BMI, height, and weight as a general health indicator
        # review with the team
        physical_columns = ['Physical-BMI', 'Physical-Height', 'Physical-Weight']
        existing_physical_columns = [col for col in physical_columns if col in self.features.columns]
        if existing_physical_columns:
            self.features['Physical_Health_Summary'] = self.features[existing_physical_columns].mean(axis=1)

    def _simplify_seasonal_features(self):
        """Simplify seasonal features into broader time-based categories."""
        # Re-maped seasonal labels: Winter, Spring, Summer, Fall to broader categories using a season_mapping dictionary.
        season_mapping = {
            'Winter': 'Cold_Season',
            'Spring': 'Warm_Season',
            'Summer': 'Warm_Season',
            'Fall': 'Cold_Season'
        }
        season_cols = [col for col in self.features.columns if 'Season' in col]
        for col in season_cols:
            if col in self.features.columns:
                self.features[col] = self.features[col].map(season_mapping).fillna('Unknown_Season')

        # Convert the new seasonal categories to one-hot encoded columns  for a more suitable format for machine
        # learning models.
        season_dummies = pd.get_dummies(self.features[season_cols], prefix=season_cols, drop_first=True)
        self.features = pd.concat([self.features.drop(columns=season_cols), season_dummies], axis=1)

    def _one_hot_encode_features(self):
        """Convert categorical features to one-hot encoded representation."""
        if len(self.categorical_columns) > 0:
            # Only attempt one-hot encoding if there are categorical columns remaining
            ohe_columns = pd.get_dummies(self.features[self.categorical_columns], drop_first=False, dtype=float)
            self.features = pd.concat([self.features.drop(columns=self.categorical_columns), ohe_columns], axis=1)
        else:
            print("No categorical columns left for one-hot encoding.")

    def _one_hot_encode_labels(self):
        """Convert labels to one-hot encoded vectors."""
        self.label.fillna(0, inplace=True)
        if self.training:
            self.one_hot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
            label_encoded = self.one_hot_encoder.fit_transform(self.label.values.reshape(-1, 1))
        else:
            label_encoded = self.one_hot_encoder.transform(self.label.values.reshape(-1, 1))

        self.label = pd.DataFrame(label_encoded, columns=[f"sii_{int(i)}" for i in range(4)])
