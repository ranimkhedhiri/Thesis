import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# Charger les données stationnaires
df = pd.read_csv(r'C:\Users\kedhi\Documents\combined_data_stationnaire.csv')

# Supprimer les lignes contenant des valeurs NaN
df_clean = df[['Close_diff', 'sum_sentiment_diff']].dropna()

# Appliquer le test de causalité de Granger
gc_result = grangercausalitytests(df_clean, maxlag=7, verbose=True)


