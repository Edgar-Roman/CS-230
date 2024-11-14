import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense, Input
from tqdm import tqdm

class Preprocessing:
    def __init__(self, df, label_col, training=True, num_imputer=None, cat_imputer=None, one_hot_encoder=None, advanced_imputation=False):
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

        # Set attributes
        self.df = df
        self.label_col = label_col
        self.training = training
        self.advanced_imputation = advanced_imputation
        self.features = self.df.drop(columns=[label_col]) if label_col in self.df else self.df
        self.label = self.df[label_col].clip(lower=0, upper=3) if label_col in self.df else None
        
        # Identify numeric and categorical columns
        self.numeric_columns = self.features.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.features.select_dtypes(include=[object]).columns

        # Print the initial count of NaNs
        initial_nan_count = self.features[self.numeric_columns].isna().sum().sum()
        print(f"Initial NaN count in numeric columns: {initial_nan_count}")

        # Initialize imputers and encoder if not provided
        self.num_imputer = num_imputer or SimpleImputer(strategy='mean')
        self.cat_imputer = cat_imputer or SimpleImputer(strategy='most_frequent')
        self.one_hot_encoder = one_hot_encoder or OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        # Automatically run preprocessing
        self._preprocess()


    def _preprocess(self):
        """Preprocess data by imputing and encoding both labeled and unlabeled data."""
        # Impute data with the selected method
        if self.advanced_imputation:
            self._impute_data_advanced()
        else:
            self._impute_data()

        # Fallback to handle any remaining NaNs
        self._impute_remaining_nans()

        # Handle outliers
        self._cap_outliers()

        # Feature engineering
        self._create_derived_features()

        # One-hot encode features
        self._one_hot_encode_features()

        # Separate labeled and unlabeled data for later use
        if self.training:
            # Separate rows with missing labels for self-training
            self.unlabeled_data = self.features[self.label.isna()].copy()
            # Filter labeled rows for model training
            self.features = self.features[~self.label.isna()]
            self.label = self.label[~self.label.isna()]

        # One-hot encode labels if available
        if self.label is not None and not self.label.isna().any():
            self._one_hot_encode_labels()

    def _impute_data(self):
        """Apply simple imputation to both numeric (mean) and categorical (most frequent) features."""
        # Impute numeric data
        self.features[self.numeric_columns] = self.num_imputer.fit_transform(self.features[self.numeric_columns])

        # Impute categorical data
        self.features[self.categorical_columns] = self.cat_imputer.fit_transform(self.features[self.categorical_columns])

    def _impute_data_advanced(self):
        """Apply advanced imputation using GAIN for numeric columns only."""
        X = self.features[self.numeric_columns].values
        mask = 1.0 - np.isnan(X).astype(np.float32)

        # Define GAIN hyperparameters
        hint_rate = 0.9
        alpha = 100
        epochs = 1000
        batch_size = 128

        # Define GAIN components
        def build_generator(input_dim):
            model = tf.keras.Sequential([
                Input(shape=(input_dim,)),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(input_dim, activation='sigmoid')
            ])
            return model

        def build_discriminator(input_dim):
            model = tf.keras.Sequential([
                Input(shape=(input_dim * 2,)),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(input_dim, activation='sigmoid')
            ])
            return model

        generator = build_generator(X.shape[1])
        discriminator = build_discriminator(X.shape[1])

        gen_optimizer = tf.keras.optimizers.Adam()
        disc_optimizer = tf.keras.optimizers.Adam()

        for epoch in tqdm(range(epochs)):
            idx = np.random.permutation(len(X))
            for start in range(0, len(X), batch_size):
                end = start + batch_size
                batch = X[idx[start:end]]
                mask_batch = mask[idx[start:end]]
                noise = np.random.uniform(0, 0.01, size=batch.shape)

                # Generate hints
                hint = np.random.binomial(1, hint_rate, size=mask_batch.shape)
                hint_batch = mask_batch * hint

                # Generator forward pass
                with tf.GradientTape() as tape:
                    imputed_data = generator(batch + noise)
                    d_input = tf.concat([imputed_data, hint_batch], axis=1)
                    d_pred = discriminator(d_input)
                    d_loss = -tf.reduce_mean(mask_batch * tf.math.log(d_pred + 1e-8) + (1.0 - mask_batch) * tf.math.log(1 - d_pred + 1e-8))
                d_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
                disc_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

                # Discriminator forward pass
                with tf.GradientTape() as tape:
                    imputed_data = generator(batch + noise)
                    g_input = tf.concat([imputed_data, mask_batch], axis=1)
                    g_pred = discriminator(g_input)
                    g_loss = tf.reduce_mean((mask_batch * (batch - imputed_data) ** 2) * alpha + (1.0 - mask_batch) * (1 - g_pred) ** 2)
                g_gradients = tape.gradient(g_loss, generator.trainable_variables)
                gen_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

        # Impute missing values with the trained generator
        X_imputed = generator(X).numpy()
        X_imputed = np.where(mask, X, X_imputed)
        self.features[self.numeric_columns] = X_imputed

        # Print statement to check remaining NaNs after GAIN
        remaining_nans_after_gain = np.isnan(X_imputed).sum()
        print(f"Remaining NaNs after GAIN imputation: {remaining_nans_after_gain}")


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

    # Feature Engineering
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

    def _impute_remaining_nans(self):
        """Fallback imputer for any remaining NaN values."""
        fallback_imputer = SimpleImputer(strategy="mean")
        self.features[self.numeric_columns] = fallback_imputer.fit_transform(self.features[self.numeric_columns])
        
        # Check remaining NaNs after fallback imputation
        remaining_nans_after_fallback = self.features[self.numeric_columns].isna().sum().sum()
        print(f"Remaining NaNs after fallback imputation: {remaining_nans_after_fallback}")

    def _one_hot_encode_features(self):
        """Convert categorical features to one-hot encoded representation."""
        ohe_columns = pd.get_dummies(self.features[self.categorical_columns], drop_first=False, dtype=float)
        self.features = pd.concat([self.features.drop(columns=self.categorical_columns), ohe_columns], axis=1)

    def _one_hot_encode_labels(self):
        """Convert labels to one-hot encoded vectors."""
        self.label.fillna(0, inplace=True)
        if self.training:
            self.label = pd.DataFrame(self.one_hot_encoder.fit_transform(self.label.values.reshape(-1, 1)),
                                      columns=[f"sii_{int(i)}" for i in range(4)])
        else:
            self.label = pd.DataFrame(self.one_hot_encoder.transform(self.label.values.reshape(-1, 1)),
                                      columns=[f"sii_{int(i)}" for i in range(4)])

    def get_labeled_data(self):
        return self.features, self.label

    def get_unlabeled_data(self):
        return self.unlabeled_data