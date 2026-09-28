import pandas as pd
from sklearn.model_selection import train_test_split

# Adapte ce chemin vers ton fichier CSV original (celui de 624 Mo)
CHEMIN_DATASET_BRUT = r"C:\Users\Lenovo\Desktop\projet 12\dataset (1).csv"
CHEMIN_SORTIE = "data/dataset_clean.csv.gz"
TAILLE_ECHANTILLON = 230_000  # nombre de lignes qu'on garde, pour rester sous 100 Mo une fois compressé

print("Chargement du dataset brut...")
df = pd.read_csv(CHEMIN_DATASET_BRUT, low_memory=False)

# On garde seulement les réclamations qui ont un texte
df_clean = df[df["Consumer Claim"].notna()]
df_clean = df_clean.drop_duplicates(subset="Consumer Claim")

# Regroupement des catégories proches (identique à ton notebook)
regroupement = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit card or prepaid card": "Credit card",
    "Prepaid card": "Credit card",
    "Checking or savings account": "Bank account or service",
    "Payday loan, title loan, or personal loan": "Payday loan",
    "Money transfer, virtual currency, or money service": "Money transfers",
    "Virtual currency": "Money transfers",
}
df_clean["Tag"] = df_clean["Tag"].replace(regroupement)

# On ne garde que les 2 colonnes utiles pour l'entraînement
df_final = df_clean[["Consumer Claim", "Tag"]]

# Échantillon stratifié : on garde les mêmes proportions de catégories, mais moins de lignes au total
df_sample, _ = train_test_split(
    df_final,
    train_size=TAILLE_ECHANTILLON,
    stratify=df_final["Tag"],
    random_state=42,
)

print(f"Lignes conservées (échantillon) : {df_sample.shape[0]}")

# Export compressé (gzip)
df_sample.to_csv(CHEMIN_SORTIE, index=False, compression="gzip")
print(f"Fichier écrit : {CHEMIN_SORTIE}")
