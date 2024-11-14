import pandas as pd

# Charge le fichier CSV avec les données combinées
data = pd.read_csv('C:/Users/kedhi/Documents/combined_data_normalisé.csv')

# Affiche les premières lignes pour vérifier
print(data.head())
# Convertir la colonne 'Date' en datetime
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Sélectionne uniquement les colonnes pertinentes pour le modèle VAR
var_data = data[['Open', 'High', 'Low', 'Close',  'sum_sentiment']]  # Ajoute d'autres colonnes si nécessaire

from statsmodels.tsa.stattools import adfuller

# Fonction pour tester la stationnarité
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Test de stationnarité pour chaque colonne
for column in var_data.columns:
    print(f'Test de stationnarité pour {column}:')
    test_stationarity(var_data[column])
var_data_diff = var_data.diff().dropna()
# Vérifier la stationnarité après différenciation
for column in var_data_diff.columns:
    print(f'Test de stationnarité pour {column} après différenciation:')
    test_stationarity(var_data_diff[column])
import pandas as pd
from statsmodels.tsa.api import VAR

# Supposons que var_data_diff soit un DataFrame contenant toutes les séries stationnaires
var_data_stationnaire = var_data_diff.dropna()

# Ajuster le modèle VAR
model = VAR(var_data_stationnaire)
results = model.fit(maxlags=15, ic='aic')  # Ajuster le modèle avec un maximum de 15 lags, en utilisant le critère AIC

# Afficher les résultats
print(results.summary())

# Faire des prévisions
n_forecast = 5  # Nombre de périodes à prévoir
forecast = results.forecast(var_data_stationnaire.values[-results.k_ar:], steps=n_forecast)

# Convertir les prévisions en DataFrame
forecast_df = pd.DataFrame(forecast, index=pd.date_range(start=var_data_stationnaire.index[-1] + pd.Timedelta(days=1), periods=n_forecast, freq='D'), columns=var_data_stationnaire.columns)
print(forecast_df)

# Appliquer l'analyse de sentiments à chaque titre
df['sentiment_score'] = df['title'].apply(sentiment_analysis)

# Regrouper les scores de sentiment par jour (somme des scores)
daily_sentiment_sum = df.groupby(df['created'].dt.date)['sentiment_score'].sum().reset_index(name='sum_sentiment')

# Afficher le résultat final
print(daily_sentiment_sum)

# Sauvegarder le résultat dans un fichier CSV
output_path = r'C:\Users\kedhi\Documents\daily_sentiment_sum.csv'
daily_sentiment_sum.to_csv(output_path, index=False)