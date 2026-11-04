"""Étape C : prétraitements linguistiques avec spaCy.

Une seule passe spaCy par avis, d'où l'on tire quatre versions du texte pour l'étude d'ablation.
Chaque version ajoute un traitement à la précédente :

    text_lemmes : lemmes, mots vides par défaut de spaCy
    text_stop   : lemmes, mots vides sur mesure
    text_neg    : + négation ("pas propre" devient NEG_propre)
    text_pos    : + filtre grammatical (noms, adjectifs, verbes, adverbes)

Dans les quatre versions, les prénoms sont remplacés par HOTE et les jetons émoji sont gardés.
"""
import re
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS

# le parser (analyse syntaxique) ne sert pas ; la NER sert à repérer les prénoms
nlp = spacy.load('fr_core_news_md', disable=['parser'])

# mots de la liste de spaCy qu'on garde : ils changent le sens d'un avis
# (négations, intensifieurs, opposition, et "proche" qui décrit l'emplacement)
INTENSIFIEURS = {'très', 'tres', 'peu', 'assez', 'plutôt', 'tellement', 'plus', 'moins'}   # "tres" : faute fréquente
A_GARDER = INTENSIFIEURS | {'ne', "n'", 'pas', 'ni', 'sans',
                            'mais', 'cependant', 'malgré',
                            'proche', 'suffisant', 'juste'}

# mots du domaine présents dans un grand nombre d'avis (voir notebook 03) :
# ils nomment le sujet de l'avis mais ne disent rien de sa qualité
MOTS_DOMAINE = {'logement', 'séjour', 'appartement', 'appart', 'hôte', 'hote', 'bordeaux', 'airbnb'}

MOTS_VIDES = (STOP_WORDS - A_GARDER) | MOTS_DOMAINE

# mots qui ouvrent une négation : elle porte sur le premier mot plein qui suit (voir variantes),
# en traversant les intensifieurs ("pas très propre" devient NEG_très NEG_propre),
# et s'arrête à une ponctuation ou à un mot de FIN_NEGATION
NEGATIONS = {'ne', "n'", 'pas', 'jamais', 'rien', 'sans', 'aucun', 'aucune', 'ni'}
FIN_NEGATION = {'mais', 'et', 'ou', 'car', 'donc'}

POS_GARDES = {'NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV'}


def mots_du_nom(nom):
    """Mots du nom d'un hôte : "Thomas Et Charlotte" devient {'thomas', 'charlotte'}."""
    if not isinstance(nom, str):
        return set()
    mots = set(re.findall(r'\w+', nom.lower()))
    # on écarte les mots courts ou courants (hôtes nommés "Appart'City", "Home Bordeaux"...)
    return {m for m in mots if len(m) >= 3 and m not in STOP_WORDS and m not in MOTS_DOMAINE}


def est_mot(token):
    """Vrai pour un mot ou un jeton émoji ; faux pour la ponctuation, les nombres et URLEXPR."""
    if token.lower_.startswith('emoji_') or token.lower_ == "n'":   # "n'" n'est pas alphabétique
        return True
    return token.is_alpha and token.text != 'URLEXPR'


def forme(token, mots_hote):
    """Ce qu'on garde du mot : HOTE pour un prénom, le jeton émoji tel quel, sinon le lemme."""
    # prénom : entité PER de spaCy, ou mot du nom de l'hôte (la NER rate certains prénoms).
    # on exige PROPN : la NER prend parfois "Bonne" ou "Logement" en début de phrase pour un prénom
    if (token.ent_type_ == 'PER' and token.pos_ == 'PROPN') or token.lower_ in mots_hote:
        return 'HOTE'
    if token.lower_.startswith('emoji_'):
        return token.lower_   # sinon spaCy le lemmatise ("emoji_visage_sourier")
    return token.lemma_.lower()


def est_mot_vide(token):
    return token.lower_ in MOTS_VIDES or token.lemma_.lower() in MOTS_VIDES


def est_negation(token):
    """Vrai pour un mot de négation ; faux pour le "pas" de "à deux pas du centre"."""
    if token.i > 0 and token.doc[token.i - 1].lower_ in ('deux', 'quelques'):
        return False
    return token.lower_ in NEGATIONS


def variantes(doc, mots_hote=()):
    """Les quatre versions du texte d'un avis analysé par spaCy."""
    lemmes, stop, neg, pos = [], [], [], []
    nier = False      # vrai tant que la négation n'a pas porté sur un mot
    apres_ne = False  # vrai après un "ne" : le "pas" qui suit ne relance pas une négation

    for token in doc:
        if token.is_punct or token.lower_ in FIN_NEGATION:
            nier = apres_ne = False
        if not est_mot(token):
            continue
        mot = forme(token, mots_hote)

        if not token.is_stop:
            lemmes.append(mot)
        vide = est_mot_vide(token)
        if not vide:
            stop.append(mot)

        # la négation est transférée aux mots suivants : "pas" lui-même n'est pas gardé
        if est_negation(token):
            # "ne correspond pas à l'annonce" : "ne" a déjà nié "correspond",
            # le "pas" ferme la négation au lieu d'en ouvrir une nouvelle
            if apres_ne and not nier:
                apres_ne = False
            else:
                nier = True
                apres_ne = token.lower_ in ('ne', "n'")
            continue
        # la négation porte sur le premier mot plein (nom, adjectif, verbe, adverbe), même s'il est
        # un mot vide : dans "rien à dire", c'est "dire" qui la prend. On saute les mots-outils
        # ("pas de problème", "n'était pas propre") et le "tout" de "pas du tout".
        if nier and token.pos_ in POS_GARDES and token.lower_ != 'tout':
            mot = 'NEG_' + mot
            nier = token.lower_ in INTENSIFIEURS   # "pas très propre" : on continue jusqu'à "propre"
        if vide:
            continue
        neg.append(mot)
        if token.pos_ in POS_GARDES or 'HOTE' in mot or 'emoji_' in mot:
            pos.append(mot)

    return {'text_lemmes': ' '.join(lemmes), 'text_stop': ' '.join(stop),
            'text_neg': ' '.join(neg), 'text_pos': ' '.join(pos)}


def pretraiter_texte(texte, mots_hote=()):
    """Les quatre versions pour un seul avis (pour tester sur un exemple)."""
    return variantes(nlp(texte), mots_hote)


def pretraiter_corpus(textes, hotes, batch_size=500):
    """Les quatre versions pour une liste d'avis, en une seule passe spaCy.

    textes : avis nettoyés AVEC majuscules (la NER s'en sert pour repérer les prénoms)
    hotes  : pour chaque avis, les mots du nom de son hôte (voir mots_du_nom)
    """
    docs = nlp.pipe(textes, batch_size=batch_size)
    return [variantes(doc, h) for doc, h in zip(docs, hotes)]
