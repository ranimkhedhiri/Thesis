import pandas as pd

# Charger les données historiques
historical_data = pd.read_csv(r'C:/Users/kedhi/mémoire/bitcoin_historical_2022.csv')

# Convertir la colonne 'Date' en datetime sans fuseau horaire
historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.tz_localize(None)

# Charger les données de sentiment
daily_sentiment_avg = pd.read_csv(r'C:\Users\kedhi\Documents\daily_sentiment_avg.csv')

# Convertir la colonne 'created' en datetime sans fuseau horaire
daily_sentiment_avg['created'] = pd.to_datetime(daily_sentiment_avg['created']).dt.tz_localize(None)

# Fusionner les DataFrames sur les colonnes 'Date' et 'created'
merged_data = pd.merge(historical_data, daily_sentiment_avg, left_on='Date', right_on='created', how='inner')

# Sélectionner uniquement les colonnes souhaitées
merged_data = merged_data[['Date', 'Volume', 'avg_sentiment', 'Close']]

# Sauvegarder le DataFrame mis à jour dans un fichier CSV
merged_data.to_csv(r'C:\Users\kedhi\Documents\merged_data.csv', index=False)

# Afficher les colonnes et les premières lignes pour vérification
print("Selected columns in the merged data:")
print(merged_data.columns)
print(merged_data.head())





