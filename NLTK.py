import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.decomposition import NMF

# Charger le fichier Excel
file_path = r'C:\Users\kedhi\Documents\posts6.xlsx'
df = pd.read_excel(file_path)

# Pré-traitement des titres des posts
# Suppression des caractères spéciaux, chiffres et transformation en minuscules
df['processed_titles'] = df['title'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x).lower()))

# Tokenisation et suppression des stop-words
stop_words = ENGLISH_STOP_WORDS  # Vous pouvez aussi ajouter vos propres stop-words
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=2000)  # 2000 mots les plus fréquents
X = vectorizer.fit_transform(df['processed_titles'])


