# Groceries-Market-Basket-Analysis
Identifying patterns of co-occurrence among items frequently purchased together by customers

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load the dataset
df = pd.read_csv('groceries.csv')

# Step 2: Data Cleaning
# Drop all columns with less than 10 non-null entries
df = df.dropna(axis=1, thresh=10)

# Drop all rows with any missing values
df = df.dropna(axis=0)

# Step 3: Transform the dataset into a list of transactions
# Convert each row into a list of items purchased
transactions = df.values.tolist()

# Step 4: Data Transformation to Binary Format
# Create a DataFrame to hold the binary representation
# One row per transaction and one column per item (with 1 indicating presence and 0 indicating absence)
all_items = set(item for sublist in transactions for item in sublist)
df_binary = pd.DataFrame(columns=all_items)

# Convert each transaction into a binary row
for transaction in transactions:
    binary_row = {item: 1 if item in transaction else 0 for item in all_items}
    df_binary = df_binary.append(binary_row, ignore_index=True)

# Step 5: Apriori Algorithm - Experiment 1 with min_support 0.005
# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets_1 = apriori(df_binary, min_support=0.005, use_colnames=True)

# Generate association rules for Experiment 1 with min confidence 0.6
association_rules_1 = association_rules(frequent_itemsets_1, metric="lift", min_threshold=1.0)
association_rules_1 = association_rules_1[association_rules_1['confidence'] >= 0.6]

# Display results for Experiment 1
print("Frequent Itemsets with min support 0.005:")
print(frequent_itemsets_1)
print("Association Rules with min confidence 0.6:")
print(association_rules_1)

# Step 6: Apriori Algorithm - Experiment 2 with min_support 0.008
# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets_2 = apriori(df_binary, min_support=0.008, use_colnames=True)

# Generate association rules for Experiment 2 with min confidence 0.3
association_rules_2 = association_rules(frequent_itemsets_2, metric="lift", min_threshold=1.0)
association_rules_2 = association_rules_2[association_rules_2['confidence'] >= 0.3]

# Display results for Experiment 2
print("Frequent Itemsets with min support 0.008:")
print(frequent_itemsets_2)
print("Association Rules with min confidence 0.3:")
print(association_rules_2)

# Step 7: Apriori Algorithm - Experiment 3 with min_support 0.015
# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets_3 = apriori(df_binary, min_support=0.015, use_colnames=True)

# Generate association rules for Experiment 3 with min confidence 0.2
association_rules_3 = association_rules(frequent_itemsets_3, metric="lift", min_threshold=1.0)
association_rules_3 = association_rules_3[association_rules_3['confidence'] >= 0.2]

# Display results for Experiment 3
print("Frequent Itemsets with min support 0.015:")
print(frequent_itemsets_3)
print("Association Rules with min confidence 0.2:")
print(association_rules_3)
