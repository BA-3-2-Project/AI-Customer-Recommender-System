import pandas as pd # type: ignore

def load_and_clean_data(file_path="data/online_retail_II.xlsx"):
    df = pd.read_excel(file_path)
    print("Columns in dataset:", df.columns.tolist())

    #step 1: Standardize column names: lowercase + replace spaces with underscores
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    #step 2: drop rows with missing customer_id or description
    df = df.dropna(subset=["customer_id", "description"])

    #step 3: remove duplicates
    df = df.drop_duplicates()

    #step 4: Convert invoice date to datetime
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])

    print(f"✅ Data cleaned. Shape: {df.shape}")
    return df
