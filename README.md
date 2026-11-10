# Prédire la note d'un logement Airbnb à partir du texte des avis

Projet de fouille de textes, Master 2 MAS (SDD), Université Rennes 2.

## Problématique

Le contenu textuel des avis voyageurs permet-il de prédire si un logement Airbnb est bien ou moins bien
noté sur la **propreté**, l'**emplacement** et le **rapport qualité-prix** ? Quels prétraitements
linguistiques rendent cette prédiction plus fiable ?

## Données

[Inside Airbnb](https://insideairbnb.com/), Bordeaux (17/09/2026), licence CC BY 4.0 :

- `data/reviews.csv.gz` : 516 990 avis ;
- `data/listings.csv.gz` : les logements et leurs notes par catégorie.

## Installation

Python 3.11 ou plus récent.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (Mac/Linux : source .venv/bin/activate)
pip install -r requirements.txt
```

## Exécution

Lancer les notebooks dans l'ordre, depuis le dossier `notebooks/`. Chacun lit le fichier écrit par le précédent
dans `data/processed/`.

| Notebook | Étape | Sortie |
|---|---|---|
| `01_nettoyage.ipynb` | A. Nettoyage par regex (HTML, URL, émojis, typographie) | `reviews_clean.parquet` |
| `02_filtrage.ipynb` | B. Filtrage (doublons, longueur, français) et jointure avec les logements | `reviews_fr.parquet` |
| `03_pretraitements_linguistiques.ipynb` | C. spaCy : lemmes, mots vides sur mesure, négation, prénoms, filtre grammatical (≈ 20 min) | `reviews_ablation.parquet` |
| `04_representation_vectorielle.ipynb` | D. Un document par logement, étiquettes, représentation TF-IDF | `logements.parquet` |
| `05_methode_de_fouille.ipynb` | E. Classification, étude d'ablation, analyse | `resultats/*.csv` |

Les fonctions utilisées par les notebooks sont dans `src/`. Après une modification d'un fichier de `src/`,
redémarrer le noyau du notebook.

## Organisation

```
data/            données initiales et fichiers produits (data/processed/)
notebooks/       une étape par notebook
src/             fonctions : nettoyage, filtrage, pre_traitement, agregation, fouille
resultats/       tableaux de résultats
rapport.md       rapport : choix, justifications, problèmes rencontrés, résultats
```
