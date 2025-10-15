import pickle
import pandas as pd # type: ignore

def load_model(file_path="item_based_model.pkl"):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def recommend_items(customer_id, user_item_matrix, similarity_matrix, top_n=5):
    if customer_id not in user_item_matrix.index:
        return ["⚠️ Customer ID not found in dataset."]
    
    #step1: Get the user's purchase history
    user_purchases = user_item_matrix.loc[customer_id]
    purchased_items = user_purchases[user_purchases > 0].index.tolist()

    if not purchased_items:
        return ["⚠️ No purchase history for this customer."]

    #step2: Score items based on similarity to purchased items
    scores = similarity_matrix[purchased_items].sum(axis=1)
    scores = scores.drop(purchased_items, errors="ignore")  # exclude already purchased

    #step3: Top-N recommendations
    recommended_items = scores.sort_values(ascending=False).head(top_n).index.tolist()
    return recommended_items

def launch_chatbot():
    #step4: Load trained model
    model_data = load_model()
    user_item_matrix = model_data["user_item_matrix"]
    similarity_matrix = model_data["similarity_matrix"]

    print("🤖 Welcome to the Item Recommender Chatbot!")
    while True:
        user_input = input("Enter Customer ID (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break
        try:
            customer_id = float(user_input)
            recommendations = recommend_items(customer_id, user_item_matrix, similarity_matrix)
            print(f"🎯 Recommended items for customer {customer_id}: {recommendations}")
        except ValueError:
