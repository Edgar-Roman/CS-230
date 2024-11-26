import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import Callback
from Accalerometer_Preprocessing import Accalerometer_Preprocessing
from Preprocessing import Preprocessing


if __name__ == "__main__":

    acc_dir = '/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet'
    tab_dir = '/kaggle/input/child-mind-institute-problematic-internet-use/train.csv'

    acc_preprocessing = Accalerometer_Preprocessing(acc_dir, tab_dir, 1)
    acc_train_df, tab_train_df = acc_preprocessing.acc_train_df, acc_preprocessing.tab_train_df
    preprocessing = Preprocessing(tab_train_df, label_col='sii', training=True, advanced_imputation=False)

    X_acc_train = acc_train_df.values
    X_acc_train = X_acc_train.reshape(-1, 60000)
    
    X_tab_train = preprocessing.features.values
    y_train = preprocessing.label.values

    smote = SMOTE()
    X_tab_train, _ = smote.fit_resample(X_tab_train, y_train)

    smote2 = SMOTE()
    X_acc_train, y_train = smote2.fit_resample(X_acc_train, y_train)

    X_tab_train, X_tab_val, X_acc_train, X_acc_val, y_train, y_val = train_test_split(X_tab_train, X_acc_train, y_train, test_size=0.2, random_state=230)
        
    # Normalize the input features
    scaler = StandardScaler()
    X_tab_train = scaler.fit_transform(X_tab_train)
    X_tab_val = scaler.transform(X_tab_val)

    X_acc_train = X_acc_train.reshape(-1, 10000, 6)
    X_acc_val = X_acc_val.reshape(-1, 10000, 6)
        
    # Print data shapes
    print(f"X_tab_train shape: {X_tab_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_acc_train shape: {X_acc_train.shape}")
    print(f"X_val shape: {X_tab_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_acc_val shape: {X_acc_val.shape}")

    # Define the input for accelerometer data (LSTM model)
    input_accelerometer = Input(shape=(10000, 6))  # Accelerometer time series input
    
    # LSTM layer for accelerometer data (outputs a vector of size 20)
    lstm_out = LSTM(20,)(input_accelerometer)
    
    # Define the input for tabular data (fully connected model)
    input_tabular = Input(shape=(X_tab_train.shape[1],))  # Tabular data input (e.g., 10 features)
    
    # Fully connected layers for tabular data (outputs a vector of size 30)
    out = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(input_tabular)
    out = Dropout(0.3)(out)
    out = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(out)
    out = Dropout(0.3)(out)
    out = Dense(32, activation='relu')(out)
    
    # Concatenate the outputs of the LSTM and FC networks
    concatenated = Concatenate()([lstm_out, out])
    
    # Pass the concatenated features through three dense layers
    dense_1 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(concatenated)
    dense_2 = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(dense_1)
    dense_3 = Dense(16, activation='relu')(dense_2)
    
    # Final output layer: Output a vector of size 4
    output = Dense(4)(dense_3)
    
    # Create the model
    model = Model(inputs=[input_accelerometer, input_tabular], outputs=output)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='mse',
                    metrics=['accuracy'])

    # Define EarlyStopping and ReduceLROnPlateau callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)


    # Train the model with EarlyStopping
    history = model.fit([X_acc_train, X_tab_train], y_train,
                        epochs=250,
                        batch_size=32,
                        validation_data=([X_acc_val, X_tab_val], y_val),
                        callbacks=early_stopping)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
