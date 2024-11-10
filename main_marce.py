import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses info and warning logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN custom operations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from Preprocessing import Preprocessing
from tensorflow.keras.utils import to_categorical

def create_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(16, activation='relu'),
        Dense(4, activation='softmax')  # 4 output neurons for 4 classes
    ])
    model.compile(optimizer=Adam(learning_rate=0.00001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load and preprocess training data
    train_file_path = 'child-mind-institute-problematic-internet-use/train.csv'
    train_df = pd.read_csv(train_file_path)

    # Use Preprocessing to handle data cleaning
    preprocessing = Preprocessing(train_df, label_col='sii', training=True)
    X_train = preprocessing.features.values
    y_train = preprocessing.label.values

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=230)

    # Normalize the input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Print data shapes
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # Create and compile the model
    input_shape = X_train.shape[1]
    model = create_model(input_shape)

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model with EarlyStopping
    history = model.fit(X_train, y_train,
                        epochs=300,
                        batch_size=16,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping])

    # Plot training & validation loss and accuracy
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

    # Evaluate the model on the validation set
    final_loss, final_accuracy = model.evaluate(X_val, y_val)
    print(f"\nFinal validation accuracy: {final_accuracy}, Final validation loss: {final_loss}")

    # Load and preprocess test data
    test_file_path = 'child-mind-institute-problematic-internet-use/test.csv'
    test_df = pd.read_csv(test_file_path)
    test_preprocessing = Preprocessing(test_df, label_col='sii', training=False,
                                       num_imputer=preprocessing.num_imputer,
                                       cat_imputer=preprocessing.cat_imputer,
                                       one_hot_encoder=preprocessing.one_hot_encoder)
    X_test = test_preprocessing.features.values
    X_test = scaler.transform(X_test)  # Apply training normalization to test data

    # Generate predictions on the test data
    test_predictions = model.predict(X_test)
    test_predicted_labels = np.argmax(test_predictions, axis=1)

    # Generate submission file with id and predicted class
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sii': test_predicted_labels
    })

    # Save submission to CSV
    submission.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' generated successfully.")
