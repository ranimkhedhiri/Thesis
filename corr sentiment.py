import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Charger les données
df = pd.read_csv(r'C:\Users\kedhi\Documents\combined_data_final.csv')  # Remplace par le chemin de ton fichier

# Vérifier les colonnes disponibles
print(df.columns)

# Calculer la corrélation de Pearson
pearson_corr, pearson_p_value = pearsonr(df['sum_sentiment'], df['Close'])
print(f'Coefficient de corrélation de Pearson: {pearson_corr:.4f}, p-value: {pearson_p_value:.4f}')

# Calculer la corrélation de Spearman
spearman_corr, spearman_p_value = spearmanr(df['sum_sentiment'], df['Close'])
print(f'Coefficient de corrélation de Spearman: {spearman_corr:.4f}, p-value: {spearman_p_value:.4f}')

# Visualiser la relation avec un scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(x='sum_sentiment', y='Close', data=df)
plt.title('Relation entre le Score de Sentiment et le Prix du Bitcoin')
plt.xlabel('Sum Sentiment')
plt.ylabel('Prix du Bitcoin (Close)')
plt.grid()
plt.show()
