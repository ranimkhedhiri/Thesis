import pandas as pd

import pandas as pd

# Charger la base de données
merged_data = pd.read_csv(r'C:\Users\kedhi\Documents\merged_data.csv')

# Conserver uniquement les colonnes souhaitées
merged_data = merged_data[['Close', 'Volume', 'avg_sentiment']]

# Sauvegarder les données filtrées dans un nouveau fichier CSV
merged_data.to_csv(r'C:\Users\kedhi\Documents\merged_data_filtered.csv', index=False)


# Assurez-vous que les colonnes sont bien nommées
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Calculer la corrélation entre sum_sentiment et Close
correlation = data[['sum_sentiment', 'Close']].corr().iloc[0, 1]

# Afficher le coefficient de corrélation
print(f"Le coefficient de corrélation entre sum_sentiment et Close est : {correlation}")
