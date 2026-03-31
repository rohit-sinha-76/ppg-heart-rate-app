import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from main import load_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

print("Loading test data...")
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("Loading cached model weights...")
model = load_model('hr_model.keras')
y_pred = model.predict(X_test, verbose=0)

print("Generating visuals...")
plt.figure(figsize=(12, 6))
samples_to_plot = min(100, len(y_test))
plt.plot(y_test[:samples_to_plot], label='Actual Heart Rate', marker='o', linestyle='-', alpha=0.7)
plt.plot(y_pred[:samples_to_plot], label='Predicted Heart Rate', marker='x', linestyle='--', alpha=0.7)
plt.legend()
plt.title('Actual vs Predicted Heart Rate (Test Set Sample)')
plt.xlabel('Sample Index')
plt.ylabel('Heart Rate (BPM)')
plt.grid(True, linestyle='--', alpha=0.6)

os.makedirs('plots', exist_ok=True)
plt.savefig('plots/actual_vs_predicted.png')
print("Fast plot saved to plots/actual_vs_predicted.png.")
