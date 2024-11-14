import pandas as pd
import matplotlib.pyplot as plt

# Load the daily sentiment average data
daily_sentiment_avg = pd.read_csv(r'C:\Users\kedhi\Documents\daily_sentiment_avg.csv')

# Convert 'created' column to datetime if not already
daily_sentiment_avg['created'] = pd.to_datetime(daily_sentiment_avg['created'])

# Plot the average sentiment over time
plt.figure(figsize=(10,6))
plt.plot(daily_sentiment_avg['created'], daily_sentiment_avg['avg_sentiment'], color='blue', label='Average Sentiment')

# Add title and labels
plt.title('Daily Average Sentiment Over Time', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Sentiment', fontsize=12)
plt.xticks(rotation=45)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
