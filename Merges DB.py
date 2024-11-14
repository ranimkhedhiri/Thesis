import pandas as pd

# Load the historical data
historical_data = pd.read_csv(r'C:\Users\kedhi\mémoire\bitcoin_historical_2022.csv')

# Convert 'Date' column to datetime (and remove timezone if present)
historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.tz_localize(None)

# Load the sentiment data
daily_sentiment_avg = pd.read_csv(r'C:\Users\kedhi\Documents\daily_sentiment_avg.csv')

# Convert 'created' column to datetime
daily_sentiment_avg['created'] = pd.to_datetime(daily_sentiment_avg['created'])

# Merge the dataframes on 'Date' and 'created' columns
merged_data = pd.merge(historical_data, daily_sentiment_avg, left_on='Date', right_on='created', how='inner')

# Save the merged data to a CSV file
merged_data.to_csv(r'C:\Users\kedhi\Documents\merged_data.csv', index=False)
import pandas as pd


# Display the first few rows of the dataframe
print(merged_data.head())



