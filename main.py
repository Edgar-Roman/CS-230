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
from tensorflow.keras.regularizers import l1, l2
from Preprocessing import Preprocessing

def get_model_architecture(model_version, input_shape):
    architectures = [
        # Model 0: Base Model with One Hidden Layer
        Sequential([
            Input(shape=(input_shape,)),
            Dense(16, activation='relu'),
            Dense(4, activation='softmax')  # 4 output neurons for 4 classes
        ]),
        
        # Model 1: Two Hidden Layer Model with Dropout
        Sequential([
            Input(shape=(input_shape,)),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(4, activation='softmax')
        ]),
        
        # Model 2: Wider Network
        Sequential([
            Input(shape=(input_shape,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(4, activation='softmax')
        ]),
        
        # Model 3: Deeper Network with Batch Normalization
        Sequential([
            Input(shape=(input_shape,)),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dense(4, activation='softmax')
        ]),
        
        # Model 4: Model with L2 Regularization
        Sequential([
            Input(shape=(input_shape,)),
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(4, activation='softmax')
        ]),
        
        # Model 5: Model with Dropout and Regularization
        Sequential([
            Input(shape=(input_shape,)),
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(4, activation='softmax')
        ]),
        # Model 6: More Neurons / Layers (Increased Model Capacity)
        Sequential([
            Input(shape=(input_shape,)),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(4, activation='softmax')
        ]),
        'wide_deep'
    ]

    # https://arxiv.org/pdf/1606.07792 (PAPER LINK)
    if model_version == 'wide_deep':
        input_layer = Input(shape=(input_shape,))

        # Wide part
        wide_output = Dense(
            4,
            activation='linear',
            kernel_regularizer=l1(0.001)
        )(input_layer)

        # Deep part
        deep = Dense(
            128,
            activation='relu',
            kernel_regularizer=l2(0.001)
        )(input_layer)
        deep = BatchNormalization()(deep)
        deep = Dropout(0.5)(deep)

        deep = Dense(
            64,
            activation='relu',
            kernel_regularizer=l2(0.001)
        )(deep)
        deep = BatchNormalization()(deep)
        deep = Dropout(0.5)(deep)

        deep = Dense(
            32,
            activation='relu',
            kernel_regularizer=l2(0.001)
        )(deep)
        deep = BatchNormalization()(deep)
        deep = Dropout(0.5)(deep)

        deep_output = Dense(
            4,
            activation='linear',
            kernel_regularizer=l2(0.001)
        )(deep)

        # Combine Wide and Deep parts
        combined_output = tf.keras.layers.add([wide_output, deep_output])
        combined_output = tf.keras.layers.Activation('softmax')(combined_output)

        model = tf.keras.Model(inputs=input_layer, outputs=combined_output)
        return model
    else:
        return architectures[model_version]


def create_model(input_shape):
    model = get_model_architecture(6, input_shape)
    model.compile(optimizer=Adam(learning_rate=0.0005), 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load training data
    train_file_path = 'child-mind-institute-problematic-internet-use/train.csv'
    train_df = pd.read_csv(train_file_path)

    preprocessing = Preprocessing(train_df, label_col='sii', training=True, advanced_imputation=False)
    # X_train, y_train = preprocessing.get_labeled_data()

    X_train = preprocessing.features.values
    y_train = preprocessing.label.values

    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    extra_iteration = False

    while preprocessing.unlabeled_data.shape[0] > 0 or extra_iteration:

        # Split training data (80% training, 20% validation)
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

        # Define EarlyStopping and ReduceLROnPlateau callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        # Train the model with EarlyStopping
        history = model.fit(X_train, y_train, 
                            epochs=1000, 
                            batch_size=64, 
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

        print(preprocessing.unlabeled_data.shape)
        if preprocessing.unlabeled_data.shape[0] > 0:

            extra_iteration = True

            # Pseudo-labeling on the unlabeled data
            confidence_threshold = 0.9
            pseudo_labels_probs = model.predict(preprocessing.unlabeled_data)
            pseudo_labels = np.argmax(pseudo_labels_probs, axis=1)
            max_confidences = np.max(pseudo_labels_probs, axis=1)

            # Separate high and low confidence indices
            high_confidence_indices = np.where(max_confidences >= confidence_threshold)[0]
            low_confidence_indices = np.where(max_confidences < confidence_threshold)[0]

            # Print the indices
            print(f"High confidence indices: {high_confidence_indices}")
            print(f"Low confidence indices: {low_confidence_indices}")

            # Extract high confidence unlabeled data
            high_confidence_data = preprocessing.unlabeled_data.iloc[high_confidence_indices]
            high_confidence_labels = pseudo_labels[high_confidence_indices]

            # Convert high confidence labels to one-hot encoded format
            high_confidence_labels_one_hot = np.eye(4)[high_confidence_labels]

            # Append high-confidence pseudo-labeled data to training set
            X_train = np.vstack([X_train, high_confidence_data.values])
            y_train = np.vstack([y_train, high_confidence_labels_one_hot])

            # Remove the added high-confidence pseudo-labeled data from unlabeled set
            preprocessing.unlabeled_data = preprocessing.unlabeled_data.iloc[low_confidence_indices]
        
        else:
            extra_iteration = False
        
        break # Currently not performing self-learning because the model performance is too poor.

    # Load and preprocess test data
    test_file_path = 'child-mind-institute-problematic-internet-use/test.csv'
    test_df = pd.read_csv(test_file_path)
    test_preprocessing = Preprocessing(test_df, label_col='sii', training=False,
                                   num_imputer=preprocessing.num_imputer,
                                   cat_imputer=preprocessing.cat_imputer,
                                   one_hot_encoder=preprocessing.one_hot_encoder,
                                   advanced_imputation=True)
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
