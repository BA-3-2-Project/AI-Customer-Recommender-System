import pickle

def evaluate_model(file_path="item_based_model.pkl"):
    with open(file_path, "rb") as f:
        model_data = pickle.load(f)

    user_item_matrix = model_data["user_item_matrix"]
    similarity_matrix = model_data["similarity_matrix"]

