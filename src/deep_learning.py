import pandas as pd 
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout 
import joblib # type: ignore
from data_preprocessing import load_and_clean_data


def run_deep_learning():
    print("🔹 Training Deep Learning Recommender...")

    #step1: Load dataset
    df = load_and_clean_data()

    #step2: Normalize column names to lowercase, strip spaces
    df.columns = df.columns.str.strip().str.lower()

    #step3: Find the customer id column dynamically
    customer_col = None
    for col in df.columns:
        if "customer" in col and "id" in col:
            customer_col = col
            break

    if not customer_col:
        raise KeyError("❌ Could not find a customer id column in dataset.")

    #step4: Drop rows with missing customer IDs
    df = df.dropna(subset=[customer_col])

    #step5: Pivot to build user-item matrix
    user_item_matrix = df.pivot_table(
        index=customer_col,
        columns="description",
        values="quantity",
        aggfunc="sum",
        fill_value=0
    )

    print(f"✅ User-Item Matrix for deep learning shape: {user_item_matrix.shape}")

    #step6: Prepare training data
    X = user_item_matrix.values
    y = (X > 0).astype(int)  # Binary target: purchased (1) or not (0)

    #step7: Build simple deep learning model
    model = Sequential([
        Dense(128, activation="relu", input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(y.shape[1], activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("🔹 Training neural network...")
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    #step8: Save model & feature index
    model.save("deep_learning_model.h5")
    joblib.dump(user_item_matrix.columns.tolist(), "deep_learning_items.pkl")

    print("✅ Deep Learning Model trained and saved!")


if __name__ == "__main__":
    run_deep_learning()

