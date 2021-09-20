import pandas as pd
import numpy as np
import duckdb

df = pd.read_csv("data/HistoricalData_AMZN.csv")
df["Date"] = pd.to_datetime(df["Date"])

# those columns have dollar sign
columns_to_clean = ["Close/Last", "Open", "High", "Low"]

# remove the $ and cast them as float
for c in columns_to_clean:
    df[c] = df[c].str.replace("$", "")
    df[c] = df[c].astype(dtype=np.float)

print(df.head())
print(df.info())

# # Connect to the DuckDB database Stocks.DB
# # If the database does not exist, it will be created
conn = duckdb.connect('Stocks.DB')


# Make an in-memory view of a pandas DataFrame
# Select only Date and Close/Last
conn.register('stocks_prices_view', df[["Date", "Close/Last"]])

# Save view to table in 'Stocks.DB': This will fail is the table already exists
conn.execute('CREATE TABLE stocks_prices AS SELECT * FROM stocks_prices_view;')

# Read a database table in a pandas dataframe
_query = "SELECT * FROM stocks_prices;"
fetched_df = conn.execute(_query).fetchdf()

print(fetched_df.head())

# Delete table
conn.execute('DROP TABLE stocks_prices;')
