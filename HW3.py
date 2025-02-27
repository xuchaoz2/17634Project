import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = "data_original.csv"
data = pd.read_csv(file_path)

# Compute statistics for "passengers"
mean_passengers = data['passengers'].mean()
std_passengers = data['passengers'].std()
min_passengers = data['passengers'].min()
max_passengers = data['passengers'].max()

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram for "passengers"
ax.hist(data['passengers'], bins=20, color='blue', edgecolor='black', alpha=0.7)

# Display text for statistics (mean, std dev, range)
stats_text = f"Mean: {mean_passengers:.2f}\nStd Dev: {std_passengers:.2f}\nRange: {min_passengers:.2f} - {max_passengers:.2f}"
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.5))

# Customize plot without legends or vertical lines
ax.set_xlabel('Passengers', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Passengers with Mean & Std Dev', fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.show()
