import pandas as pd
import os

data_folder = "data"

files = [
    "restaurant.csv",
    "users.csv",
    "food.csv",
    "menu.csv",
    "orders.csv",
    "order_items.csv",
    "reviews.csv"
]

for file in files:

    path = os.path.join(data_folder, file)

    # Only read first 5 rows
    df = pd.read_csv(path, nrows=5)

    print("\n" + "=" * 60)
    print("FILE:", file)
    print("=" * 60)

    print("\nCOLUMNS:")
    print(df.columns.tolist())

    print("\nDATA TYPES:")
    print(df.dtypes)

    print("\nSAMPLE:")
    print(df.head())