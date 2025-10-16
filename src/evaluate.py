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

