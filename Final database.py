import pandas as pd

# Charger la base de données
merged_data = pd.read_csv(r'C:\Users\kedhi\Documents\merged_data.csv')

# Conserver uniquement les colonnes souhaitées
merged_data = merged_data[['Date','Close', 'Volume', 'avg_sentiment']]

# Sauvegarder les données filtrées dans un nouveau fichier CSV
merged_data.to_csv(r'C:\Users\kedhi\Documents\merged_data_filtered.csv', index=False)

