import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.preprocessing import MinMaxScaler
import joblib
from data_preprocessing import load_and_clean_data

def run_deep_learning():
    print("🔹 Training Deep Learning Recommender...")

    df = load_and_clean_data()
    df.columns = df.columns.str.strip().str.lower()

    customer_col = None
    for col in df.columns:
        if "customer" in col and "id" in col:
            customer_col = col
            break
    if not customer_col:
        raise KeyError("❌ Could not find a customer id column in dataset.")

    df = df.dropna(subset=[customer_col])

    user_item_matrix = df.pivot_table(
        index=customer_col,
        columns="description",
        values="quantity",
        aggfunc="sum",
        fill_value=0
    )

    print(f"✅ User-Item Matrix shape: {user_item_matrix.shape}")

    X = user_item_matrix.values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y = (X > 0).astype(int)

    model = Sequential([
        Dense(256, activation="relu", input_shape=(X_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(y.shape[1], activation="sigmoid")
    ])

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

    print("🔹 Training neural network...")
    model.fit(X_scaled, y, epochs=50, batch_size=64, callbacks=[early_stop], verbose=1)

    model.save("deep_learning_model.keras")
    joblib.dump(user_item_matrix.columns.tolist(), "deep_learning_items.pkl")

    print("✅ Deep Learning Model trained and saved!")

if __name__ == "__main__":
    run_deep_learning()
