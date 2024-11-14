# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the historical data
historical_data = pd.read_csv(r'C:\Users\kedhi\mémoire\bitcoin_historical_2022.csv')

# Convert 'Date' to datetime if necessary and set it as the index
historical_data['Date'] = pd.to_datetime(historical_data['Date'])
historical_data.set_index('Date', inplace=True)

# Define the variables you want to plot
variables = ["Open", "High", "Close", "Low", "Volume"]

# Create a figure with 5 subplots (1 row and 5 columns)
fig, axs = plt.subplots(len(variables), 1, figsize=(12, 15), sharex=True)

# Plot each variable in its own subplot
for i, column in enumerate(variables):
    axs[i].plot(historical_data.index, historical_data[column], label=column)
    axs[i].set_title(f'{column} Over Time')
    axs[i].set_ylabel(column)
    axs[i].legend(loc='upper right')

# Set the x-axis label for the bottom subplot only
axs[-1].set_xlabel('Date')

# Adjust layout for readability
plt.tight_layout()
plt.show()
