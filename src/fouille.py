"""Étape E : classification des logements (bien noté / moins bien noté) à partir du texte."""
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score


def pipeline(modele, ngrammes=(1, 2)):
    """TF-IDF puis modèle : le TF-IDF est appris sur les données d'entraînement seulement."""
    tfidf = TfidfVectorizer(strip_accents='unicode', min_df=5, ngram_range=ngrammes)
    return make_pipeline(tfidf, modele)


def evaluer(X, y, modele, ngrammes=(1, 2)):
    """F1 moyen en validation croisée à 5 plis (chaque logement est testé une fois)."""
    plis = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline(modele, ngrammes), X, y, cv=plis, scoring='f1_macro')
    return scores.mean()


def mots_associes(pipe, n=15):
    """Termes aux coefficients les plus forts du modèle linéaire (SVM linéaire ici), pour chaque classe."""
    tfidf, modele = pipe[0], pipe[-1]
    coefs = pd.Series(modele.coef_[0], index=tfidf.get_feature_names_out()).sort_values()
    return pd.DataFrame({
        'moins bien noté': coefs.index[:n],          # coefficients les plus négatifs : classe 0
        'mieux noté': coefs.index[::-1][:n],         # coefficients les plus positifs : classe 1
    })
