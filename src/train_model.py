import pandas as pd 
import numpy as np 
import pickle
from sklearn.metrics.pairwise import cosine_similarity 
from data_preprocessing import load_and_clean_data


#step1: train the model 
def train_item_based_model():
    df = load_and_clean_data()
    print(f"✅ Data cleaned. Shape: {df.shape}")

    #step2: Build user-item interaction matrix
    user_item_matrix = df.pivot_table(
        index="customer id", columns="description", values="quantity", aggfunc="sum", fill_value=0
    )
    print(f"User-Item Matrix shape: {user_item_matrix.shape}")

    #step3: Compute item-based similarity
    similarity = cosine_similarity(user_item_matrix.T)
    print("✅ Item-based similarity matrix trained!")

    #step4: Save both similarity AND user_item_matrix
    with open("item_based_model.pkl", "wb") as f:
        pickle.dump((similarity, user_item_matrix), f)

    print("✅ Model saved as item_based_model.pkl")

if __name__ == "__main__":
    train_item_based_model()

