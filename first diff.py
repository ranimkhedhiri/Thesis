
import pandas as pd

# Charger les données
df = pd.read_csv(r'C:\Users\kedhi\Documents\combined_data_final.csv')

# Appliquer la première différence pour stationnariser les variables
df['Close_diff'] = df['Close'].diff()
df['Volume_diff'] = df['Volume'].diff()
df['sum_sentiment_diff'] = df['sum_sentiment'].diff()

# Afficher les premières lignes après transformation
print(df[['Close_diff', 'Volume_diff', 'sum_sentiment_diff']].head())

# Enregistrer la nouvelle base stationnaire
df.to_csv(r'C:\Users\kedhi\Documents\combined_data_stationnaire.csv', index=False)

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# Charger les données stationnaires
df = pd.read_csv(r'C:\Users\kedhi\Documents\combined_data_stationnaire.csv')

# Appliquer le test de causalité de Granger
gc_result = grangercausalitytests(df[['Close_diff', 'sum_sentiment_diff']], maxlag=5, verbose=True)
