import pickle

def evaluate_model(file_path="item_based_model.pkl"):
    with open(file_path, "rb") as f:
        model_data = pickle.load(f)

    user_item_matrix = model_data["user_item_matrix"]
    similarity_matrix = model_data["similarity_matrix"]
#step 1: simple evaluation: check top 5 recommendations for first user
    first_user = user_item_matrix.index[0]
    user_ratings = user_item_matrix.loc[first_user]
    already_purchased = user_ratings[user_ratings > 0].index.tolist()
    #step 2: recommend top 5 items not already purchased
    scores = similarity_matrix.sum()
    recommendations = [item for item in scores.sort_values(ascending=False).index if item not in already_purchased][:5]

    print(f"✅ Evaluation example for user {first_user}: {recommendations}")

if __name__ == "__main__":
    evaluate_model()
