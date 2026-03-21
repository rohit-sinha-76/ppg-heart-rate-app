"""
Training script for the PPG Heart Rate Prediction Model using a 1D Convolutional Neural Network.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def load_data():
    """
    Loads PPG signals and corresponding heart rates from the BIDMC dataset.
    Extracts 8-second windows (1000 samples at 125Hz) and standardizes them.
    
    Returns:
        X (np.ndarray): Processed PPG windows, shape (samples, window_size).
        y (np.ndarray): Corresponding target heart rates, shape (samples,).
    """
    fs = 125  # Sampling rate: 125 Hz
    window_duration = 8  # Window length in seconds
    window_size = fs * window_duration  # 1000 samples per window

    all_X = []
    all_y = []

    for i in range(1, 54):
        num = str(i).zfill(2)
        signal_path = f"bidmc_csv/Signals/bidmc_{num}_Signals.csv"
        numerics_path = f"bidmc_csv/Numerics/bidmc_{num}_Numerics.csv"
        
        # Skip if files do not exist
        if not os.path.exists(signal_path) or not os.path.exists(numerics_path):
            continue

        try:
            signal_df = pd.read_csv(signal_path)
            numerics_df = pd.read_csv(numerics_path)

            ppg_values = signal_df[" PLETH"].values
            hr_values = numerics_df[" HR"].values

            # Extract windows aligned with heart rate readings (overlapping lookback windows)
            for j in range(window_duration, len(hr_values)):
                end_idx = j * fs
                start_idx = end_idx - window_size
                
                if start_idx >= 0 and end_idx <= len(ppg_values):
                    window = ppg_values[start_idx:end_idx]
                    
                    # Discard invalid or flatline windows
                    if window.std() == 0 or np.isnan(window).any() or np.isnan(hr_values[j]):
                        continue

                    # Standardize the signal window (Z-score normalization)
                    window_normalized = (window - window.mean()) / window.std()
                    
                    all_X.append(window_normalized)
                    all_y.append(hr_values[j])

        except Exception as e:
            print(f"Error processing patient {i}: {e}")

    return np.array(all_X), np.array(all_y)

def augment_data(X, y):
    """
    Augments the PPG signal data to synthetically increase dataset size.
    Applies random noise, slight scaling, and shifting to make the model
    robust against real-world sensor variations.
    """
    augmented_X, augmented_y = [], []
    
    for i in range(len(X)):
        signal, hr = X[i], y[i]
        
        # Original signal
        augmented_X.append(signal)
        augmented_y.append(hr)
        
        # 1. Add Random Gaussian Noise
        noise = np.random.normal(0, 0.05, signal.shape)
        augmented_X.append(signal + noise)
        augmented_y.append(hr)
        
        # 2. Random Amplitude Scaling (±10%)
        scale_factor = np.random.uniform(0.9, 1.1)
        augmented_X.append(signal * scale_factor)
        augmented_y.append(hr)
        
        # 3. Random Time Shift (roll the array)
        shift_amount = np.random.randint(-15, 15)
        shifted_signal = np.roll(signal, shift_amount)
        augmented_X.append(shifted_signal)
        augmented_y.append(hr)
        
    return np.array(augmented_X), np.array(augmented_y)

def build_and_train_model(X_train, y_train):
    """
    Compiles and trains an advanced 1D CNN model.
    Uses BatchNormalization and robust Dropout to prevent overfitting 
    on the augmented dataset.
    """
    # Advanced 1D CNN Architecture
    model = models.Sequential([
        # Block 1
        layers.Conv1D(64, kernel_size=7, padding='same', input_shape=(1000, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Block 2
        layers.Conv1D(128, kernel_size=5, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Block 3
        layers.Conv1D(256, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.4),
        
        # Fully Connected Classifier
        layers.Flatten(),
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.005)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(128, kernel_regularizer=regularizers.l2(0.005)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(1)  # Regression output (HR value)
    ])

    # Dynamic Learning Rate Scheduler to speed up initial training and fine-tune later
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss='huber',    # Huber loss is more robust to outliers than MSE
        metrics=['mae']  # Mean Absolute Error in BPM
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=15,    # Increased patience due to augmentation volatility
        verbose=1,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=150,      # Give it time to train on the augmented data
        batch_size=64,   # Larger batch size for stability with BatchNorm
        validation_split=0.2,
        verbose=1,
        callbacks=[early_stopping]
    )

    return model, history

def main():
    print("Loading and preprocessing data from the BIDMC dataset...")
    X, y = load_data()
    
    print(f"Base dataset loaded. Features shape: {X.shape}, Target shape: {y.shape}")
    print("Applying data augmentation to synthetically increase dataset size...")
    
    # Split dataset: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Base Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    print("Applying data augmentation ONLY to the training set...")
    
    # Apply synthetic augmentation to Training set only
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    
    # Reshape features to include the channel dimension: (Samples, Timesteps, Channels)
    X_train = X_train_aug.reshape(X_train_aug.shape[0], X_train_aug.shape[1], 1)
    y_train = y_train_aug.reshape(-1, 1)
    
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_test = y_test.reshape(-1, 1)

    print(f"Augmentation complete. New Training set size: {X_train.shape[0]} samples.")
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    print("\nStarting model training...")
    model, _ = build_and_train_model(X_train, y_train)

    print("\nEvaluating model on unseen test data...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Mean Absolute Error: {test_mae:.2f} BPM")

    # Save the trained model
    model.save('hr_model.keras')
    print("Model successfully saved as 'hr_model.keras'\n")

    # Generate predictions to visualize
    print("Generating predictions on test set...")
    y_pred = model.predict(X_test, verbose=0)

    # Visualize predictions
    plt.figure(figsize=(12, 6))
    samples_to_plot = min(100, len(y_test))
    
    plt.plot(y_test[:samples_to_plot], label='Actual Heart Rate', marker='o', linestyle='-', alpha=0.7)
    plt.plot(y_pred[:samples_to_plot], label='Predicted Heart Rate', marker='x', linestyle='--', alpha=0.7)
    
    plt.legend()
    plt.title('Actual vs Predicted Heart Rate (Test Set Sample)')
    plt.xlabel('Sample Index')
    plt.ylabel('Heart Rate (BPM)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    print("Displaying visualization... Close the plot to exit the script.")
    plt.show()

if __name__ == "__main__":
    main()
