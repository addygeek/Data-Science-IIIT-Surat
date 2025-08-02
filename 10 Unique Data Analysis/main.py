import pandas as pd

# Sample dataset: Customer transaction data
# Creating a DataFrame for demonstration
data = {
    'Transaction_ID': [101, 102, 103, 104, 105],
    'Customer_Name': ['Alice', 'Bob', 'Alice', 'David', 'Eve'],
    'Product': ['Laptop', 'Mouse', 'Laptop', 'Keyboard', 'Mouse'],
    'Amount': [1200, 25, 1200, 75, 25],
    'Category': ['Electronics', 'Accessories', 'Electronics', 'Accessories', 'Accessories']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print("Original Dataset:")
print(df)

# ===============================================
# 1. Find Unique Customers
# ===============================================
unique_customers = df['Customer_Name'].unique()
print("\nUnique Customers:")
print(unique_customers)

# ===============================================
# 2. Find Unique Products Purchased
# ===============================================
unique_products = df['Product'].unique()
print("\nUnique Products Purchased:")
print(unique_products)

# ===============================================
# 3. Count the Frequency of Each Product
# ===============================================
product_frequency = df['Product'].value_counts()
print("\nFrequency of Products Purchased:")
print(product_frequency)

# ===============================================
# 4. Identify Rare Products (Purchased Only Once)
# ===============================================
rare_products = product_frequency[product_frequency == 1].index.tolist()
print("\nRare Products (Purchased Only Once):")
print(rare_products)

# ===============================================
# 5. Group Data by Category and Summarize
# ===============================================
category_summary = df.groupby('Category').agg({
    'Amount': ['sum', 'mean'],  # Total and Average Amount per Category
    'Transaction_ID': 'count'   # Total Transactions per Category
})
print("\nCategory Summary (Total, Average Amount, Transactions):")
print(category_summary)

# ===============================================
# 6. Find Unique Combinations of Customer and Product
# ===============================================
unique_combinations = df[['Customer_Name', 'Product']].drop_duplicates()
print("\nUnique Customer-Product Combinations:")
print(unique_combinations)

# ===============================================
# 7. Advanced: Identify Outliers in Transaction Amounts
# ===============================================
def detect_outliers(series):
    """
    Function to detect outliers using the Interquartile Range (IQR) method.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series < lower_bound) | (series > upper_bound)]

outliers = detect_outliers(df['Amount'])
print("\nOutliers in Transaction 'l


output_file = "processed_data.csv"
df.to_csv(output_file, index=False)
print(f"\nProcessed dataset saved to {output_file}")
