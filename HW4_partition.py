import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "data_updated.csv"
data = pd.read_csv(file_path)

# Split into training (70%) and temp (30%)
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)

# Split temp_data into validation (15%) and testing (15%)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save datasets
train_data.to_csv("data_train.csv", index=False)
val_data.to_csv("data_validation.csv", index=False)
test_data.to_csv("data_test.csv", index=False)

print("Data partitioned into training (70%), validation (15%), and testing (15%) datasets.")
