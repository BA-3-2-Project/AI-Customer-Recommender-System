import pandas as pd   # type: ignore
import matplotlib.pyplot as plt # type: ignore
from data_preprocessing import load_and_clean_data  # type: ignore

def analyze_sales_over_time():
    df = load_and_clean_data()
    df.set_index('invoicedate', inplace=True)
    monthly_sales = df.groupby(pd.Grouper(freq='M'))['quantity'].sum()
    print("Monthly sales sample:\n", monthly_sales.head())

    monthly_sales.plot(figsize=(10,5))
    plt.title("Monthly Sales Quantity Over Time")
    plt.xlabel("Month")
    plt.ylabel("Quantity Sold")
    plt.show()

if __name__ == "__main__":
    analyze_sales_over_time()
