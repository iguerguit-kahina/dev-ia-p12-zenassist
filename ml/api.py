import pickle  # pour recharger le modèle sauvegardé (model.pkl)

from fastapi import FastAPI        # pour créer notre application API
from pydantic import BaseModel     # pour décrire la forme des données reçues (user_claim)

# On ouvre et on charge le modèle UNE SEULE FOIS, au démarrage du serveur
# "rb" = lecture en mode binaire (obligatoire pour pickle, comme "wb" pour l'écriture)
with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)  # on déballe le colis (l'inverse de pickle.dump)

# On récupère séparément les 3 éléments qu'on avait emballés ensemble
model = bundle["model"]
vectorizer = bundle["vectorizer"]
label_encoder = bundle["label_encoder"]

# On crée notre application FastAPI, à laquelle on va ajouter des routes juste après
app = FastAPI()

# On décrit la forme des données qu'on attend dans la requête POST
class Reclamation(BaseModel):
    user_claim: str

# On définit la route /tags, qui accepte une requête de type POST
@app.post("/tags")
def predict_tag(reclamation: Reclamation):
    # On transforme le texte reçu en nombres, avec le même vectorizer
    # que celui utilisé pendant l'entraînement
    texte_vectorise = vectorizer.transform([reclamation.user_claim])

    # Le modèle prédit un numéro de catégorie (pas encore le nom lisible)
    prediction_numero = model.predict(texte_vectorise)

    # On retransforme ce numéro en nom de catégorie lisible
    # (l'inverse de ce que fait LabelEncoder à l'entraînement)
    prediction_categorie = label_encoder.inverse_transform(prediction_numero)

    # On renvoie le résultat au frontend
    return {"tag": prediction_categorie[0]}