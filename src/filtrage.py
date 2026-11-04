"""Étape B : fonctions de filtrage des avis nettoyés (longueur et langue)."""
import re
from lingua import LanguageDetectorBuilder

# jetons ajoutés à l'étape A : ce ne sont pas des mots de l'avis
JETONS = re.compile(r'emoji_\w+|URLEXPR')


def retirer_jetons(text):
    """Retire les jetons emoji_.. et URLEXPR (sinon « great stay emoji_visage_souriant » ressemble à du français)."""
    return JETONS.sub(' ', text)


def compter_mots(text):
    """Nombre de mots de l'avis, sans compter les jetons."""
    return len(retirer_jetons(text).split())


def detecter_langues(textes):
    """Langue de chaque texte (ex. 'FRENCH'), ou None si lingua ne sait pas."""
    detecteur = LanguageDetectorBuilder.from_all_languages().build()

    langues = detecteur.detect_languages_in_parallel_of([retirer_jetons(t) for t in textes])
    return [langue.name if langue else None for langue in langues]
