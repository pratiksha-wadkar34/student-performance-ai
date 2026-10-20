import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load Dataset
print("Loading dataset...")

df = pd.read_csv("student-mat.csv", sep=";")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Select Input Features
features = [
    "age",
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2"
]

X = df[features].copy()
y = df["G3"].copy()

# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Create Multilayer Neural Network
model = Sequential([
    Input(shape=(len(features),)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="linear")
])

# 6. Compile Model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# 7. Early Stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

# 8. Train Model
print("\nTraining Neural Network...")

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)

# 9. Evaluate Model
predictions = model.predict(X_test_scaled).flatten()

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("----------------")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 10. Save Model and Scaler
os.makedirs("models", exist_ok=True)

model.save("models/student_model.keras")

with open("models/scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

with open("models/features.pkl", "wb") as file:
    pickle.dump(features, file)

print("\nModel training completed!")
print("Saved model: models/student_model.keras")
print("Saved scaler: models/scaler.pkl")
print("Saved features: models/features.pkl")