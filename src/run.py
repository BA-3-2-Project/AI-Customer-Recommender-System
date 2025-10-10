import os
import pickle
import pandas as pd # type: ignore
from data_preprocessing import load_and_clean_data
from train_model import train_item_based_model
from evaluate import evaluate_model
from chatbot import launch_chatbot
from deep_learning import run_deep_learning
from time_series import analyze_sales_over_time

print("🚀 Starting Full AI Customer Recommendation System Pipeline...\n")

# 🔹 Step 1: Preprocessing + Time Series
print("🔹 Step 1: Time Series Analysis...")
df = load_and_clean_data()
analyze_sales_over_time()

# 🔹 Step 2: Check or Train Recommender Model
print("\n🔹 Step 2: Train / Load ML Recommender...")
if not os.path.exists("item_based_model.pkl"):
    print("⚠️ No model found. Training a new recommender model...")
    train_item_based_model()
else:
    print("✅ Found existing recommender model. Using it.")

# 🔹 Step 3: Evaluate ML Model
print("\n🔹 Step 3: Evaluate ML Model...")
evaluate_model()

# 🔹 Step 4: Run Deep Learning Model
print("\n🔹 Step 4: Running Deep Learning Model...")
run_deep_learning()

# 🔹 Step 5: Launch Chatbot
print("\n🔹 Step 5: Launch Chatbot...")
launch_chatbot()

