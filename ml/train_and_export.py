import json      # pour écrire le fichier metrics.json à la fin
import pickle     # pour sauvegarder le modèle entraîné en fichier .pickle
import time       # pour mesurer le temps d'entraînement

import pandas as pd  # pour charger et manipuler le dataset

from sklearn.feature_extraction.text import TfidfVectorizer  # transforme le texte en nombres
from sklearn.linear_model import LogisticRegression          # l'algorithme de classification choisi
from sklearn.metrics import classification_report, f1_score  # pour évaluer les performances du modèle
from sklearn.model_selection import train_test_split         # pour séparer entraînement / test
from sklearn.preprocessing import LabelEncoder                # transforme les catégories (texte) en nombres

# Chemins des fichiers : où on lit les données, où on écrit les résultats
CHEMIN_DATASET = "data/dataset_clean.csv.gz"
CHEMIN_MODELE = "model.pkl"
CHEMIN_METRIQUES = "metrics.json"

print("Chargement du dataset...")
df = pd.read_csv(CHEMIN_DATASET)  # pandas décompresse le .gz automatiquement

X = df["Consumer Claim"]  # le texte des réclamations (ce qu'on donne au modèle en entrée)
y = df["Tag"]              # la catégorie à prédire (ce que le modèle doit deviner)

# Split : 80% pour entraîner le modèle, 20% pour le tester ensuite (identique à ton notebook)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Vectorisation TF-IDF ---
# Le modèle ne comprend pas le texte, il faut le transformer en nombres
print("Vectorisation TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)      # on garde les 5000 mots les plus importants
X_train_tfidf = vectorizer.fit_transform(X_train)    # apprend le vocabulaire + transforme le train
X_test_tfidf = vectorizer.transform(X_test)          # transforme le test avec le même vocabulaire

# --- Encodage des catégories ---
# Le modèle a aussi besoin que les catégories (texte) soient des nombres
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)  # apprend les catégories + les transforme
y_test_encoded = label_encoder.transform(y_test)         # transforme le test avec les mêmes numéros

# --- Entraînement du modèle ---
print("Entraînement du modèle...")
debut = time.time()                                   # heure de départ, pour mesurer la durée
model = LogisticRegression(max_iter=1000)             # crée l'algorithme
model.fit(X_train_tfidf, y_train_encoded)             # il apprend à partir des données d'entraînement
duree_entrainement = time.time() - debut              # durée totale de l'entraînement

# --- Évaluation ---
predictions = model.predict(X_test_tfidf)  # le modèle prédit les catégories du jeu de test
f1 = f1_score(y_test_encoded, predictions, average="weighted")  # score global de performance
rapport = classification_report(  # détail des performances par catégorie
    y_test_encoded, predictions, target_names=label_encoder.classes_, output_dict=True
)

print(f"F1-score (pondéré) : {f1:.3f}")

# --- Export du modèle ---
# On regroupe modèle + vectorizer + label_encoder ensemble : les 3 sont indispensables
# pour pouvoir refaire une prédiction plus tard (sans eux, le modèle seul ne sert à rien)
bundle = {
    "model": model,
    "vectorizer": vectorizer,
    "label_encoder": label_encoder,
}
with open(CHEMIN_MODELE, "wb") as f:  # "wb" = écriture en mode binaire (obligatoire pour pickle)
    pickle.dump(bundle, f)
print(f"Modèle exporté : {CHEMIN_MODELE}")

# --- Export des métriques ---
# Un résumé des performances du modèle, lisible sans avoir à tout réentraîner
metriques = {
    "f1_score_weighted": round(f1, 4),
    "training_duration_seconds": round(duree_entrainement, 2),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "classification_report": rapport,
}
with open(CHEMIN_METRIQUES, "w") as f:
    json.dump(metriques, f, indent=2)
print(f"Métriques exportées : {CHEMIN_METRIQUES}")