import { NextResponse } from 'next/server';
import { generateText } from 'ai';
import { mistral } from '@ai-sdk/mistral';

// Le dictionnaire de catégories : exactement le même que celui du notebook
// de la mission 1 (juste traduit en objet JavaScript au lieu d'un dict Python)
const descriptionsCategories = {
    "Credit reporting": "problèmes liés aux rapports de solvabilité (erreurs, litiges sur le dossier de crédit)",
    "Debt collection": "recouvrement de créances par une agence ou un créancier",
    "Mortgage": "prêt immobilier (achat ou refinancement de logement)",
    "Credit card": "carte de crédit (frais, litiges sur transactions, cashback)",
    "Bank account or service": "compte bancaire courant ou épargne, et services associés",
    "Student loan": "prêt étudiant",
    "Consumer Loan": "prêt personnel, non lié à un véhicule ou un logement",
    "Vehicle loan or lease": "prêt ou location spécifique à un véhicule (voiture, moto)",
    "Money transfers": "transfert d'argent ou service monétaire",
    "Payday loan": "prêt sur salaire à court terme",
    "Other financial service": "autre service financier ne correspondant à aucune catégorie ci-dessus",
};

// Cette fonction POST s'exécute automatiquement quand quelqu'un appelle
// l'adresse /api/classify avec la méthode POST (c'est ce que fait le fetch()
// dans Claim.jsx quand on clique sur le bouton Catégorie suggérée)
export async function POST(request) {
    try {
        //----------------Récupration----------------
        // On récupère le texte de la réclamation envoyé par Claim.jsx
        // (le "content" qu'on avait mis dans le fetch)
        const { content } = await request.json();

        // Sécurité : si jamais aucun texte n'a été envoyé, on arrête tout de suite
        // et on renvoie une erreur claire (code 400 = "mauvaise demande")
        if (!content) {
            return NextResponse.json({ error: 'Contenu manquant' }, { status: 400 });
        }

        // On transforme le dictionnaire de catégories en texte lisible,
        // ligne par ligne, pour l'insérer dans le prompt

        const listeCategories = Object.entries(descriptionsCategories)
            .map(([cat, desc]) => `- ${cat} : ${desc}`)
            .join('\n');


        //-------------Construction Prompt------------
        // Le prompt : exactement la même structure que dans notebook,

        const prompt = `Tu es un assistant qui classe des réclamations clients dans une catégorie.
Voici les catégories possibles, avec leur description :
${listeCategories}

Réclamation : "${content}"

Réponds uniquement avec le nom exact d'une catégorie de la liste, sans explication.`;

        //------------------------Envoi / appel---------------------------
        // L'appel réel à Mistral : generateText() est la fonction du AI SDK
        // (équivalent de client.chat.complete() en Python).
        // mistral('mistral-small-latest') va automatiquement chercher
        // MISTRAL_API_KEY dans .env.local pour s'authentifier
        const { text } = await generateText({
            model: mistral('mistral-small-latest'),
            prompt,
            temperature: 0, // 0 = réponses les plus stables/prévisibles possible
        });
        //----------------- Retour------------------------
        // On renvoie la catégorie trouvée (nettoyée des espaces en trop)
        // vers Claim.jsx, qui va l'afficher à l'écran
        return NextResponse.json({ category: text.trim() });
    } catch (error) {
        // Si quoi que ce soit se passe mal (Mistral injoignable, clé invalide...),
        // on l'écrit dans les logs du serveur et on renvoie une erreur générique
        console.error('Erreur de classification :', error);
        return NextResponse.json({ error: 'Échec de la classification' }, { status: 500 });
    }
}
