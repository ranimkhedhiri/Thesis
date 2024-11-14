import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Charger les données filtrées
merged_data = pd.read_csv(r'C:\Users\kedhi\Documents\merged_data_filtered.csv')

# Fonction pour appliquer le test ADF à chaque colonne
def adf_test(series):
    result = adfuller(series)
    return {'ADF Statistic': result[0], 'p-value': result[1], 'Critical Values': result[4]}

# Appliquer le test ADF à chaque colonne et stocker les résultats
adf_results = {}
for column in merged_data.columns:
    adf_results[column] = adf_test(merged_data[column])

# Afficher les résultats
for column, result in adf_results.items():
    print(f"Results for {column}:")
    print(f"ADF Statistic: {result['ADF Statistic']}")
    print(f"p-value: {result['p-value']}")
    print(f"Critical Values: {result['Critical Values']}")
    print("\n")
