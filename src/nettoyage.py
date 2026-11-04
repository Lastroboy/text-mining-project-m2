"""Étape A : fonctions de nettoyage des avis Airbnb.

Chaque fonction prend une chaîne (un avis) : on peut la tester sur un exemple isolé.
Sur les données, nettoyer() enchaîne tout en une seule passe : df['comments'].apply(nettoyer).
"""
import re
import html
import emoji

# regex compilées une seule fois (plus rapide sur 500 000 avis)

# <br>, <br/>, <br /> (159 000 avis) : balise HTML, pas du texte
BR = re.compile(r'<br\s*/?>', flags=re.IGNORECASE)

# URL avec http(s) ou www (TD5) : n'apporte rien à l'analyse
URL = re.compile(r'https?:\S+|\bwww\.\S+')

# suite d'espaces, \n, \r ou tabulations devient un seul espace
ESPACES = re.compile(r'\s+')

# pré-filtre : permet d'analyser les émojis plus rapidement (15s au lieu de 81s),
# car les émojis sont tous après U+2100 et la bibliothèque emoji est lente
# shortcut: rate © ® ‼ ⁉ (avant U+2100), retirer ce pré-filtre si on veut aussi les compter
SYMBOLE = re.compile('[' + chr(0x2100) + '-' + chr(0x10FFFF) + ']')

# couleurs de peau des émojis: 👍🏻 et 👍 donnent le même mot
PEAU = re.compile(r'[\U0001F3FB-\U0001F3FF]')

# ’ ‘ ´ devient ' et «" “ ” devient " : l’hôte devient donc l'hôte
TYPO = str.maketrans({'’': "'", '‘': "'", '´': "'", '«': '"', '»': '"', '“': '"', '”': '"'})


def est_sans_contenu(text):
    """Vrai si l'avis ne contient ni lettre, ni chiffre, ni émoji (ex. ".", "-")."""
    # [^\W_] = lettre ou chiffre : garde "10/10" et "👍"
    return re.search(r'[^\W_]', text) is None and compter_emojis(text) == 0


def remplacer_br(text):
    """Remplace <br>, <br/>, <br /> par un espace et décode les entités HTML."""
    # pas d'autre vraie balise dans le corpus : les "<côté jardin >" sont du texte, on les garde
    return html.unescape(BR.sub(' ', text))


def normaliser_espaces(text):
    """Remplace toute suite d'espaces, \\n, \\r ou tabulations par un seul espace, puis strip."""
    return ESPACES.sub(' ', text).strip()


def compter_emojis(text):
    """Nombre d'émojis dans l'avis (👍🏻 compte pour un seul)."""
    if not SYMBOLE.search(text):   # la bibliothèque emoji est lente : on l'évite si possible
        return 0
    return emoji.emoji_count(text)


def mettre_en_minuscules(text):
    """Met l'avis en minuscules."""
    return text.lower()


def nom_emoji(caracteres, infos):
    """Nom français d'un émoji en un seul mot : 😉 devient emoji_visage_faisant_un_clin_d_œil."""
    # \W (’ - espace) devient _ : "clin_d’œil" reste un seul mot
    nom = emoji.demojize(caracteres, language='fr').strip(':')
    return ' emoji_' + re.sub(r'\W', '_', nom) + ' '


def remplacer_emojis(text):
    """Remplace chaque émoji par son nom (ils portent du sens et du sentiment, on ne les supprime pas)."""
    if not SYMBOLE.search(text):
        return text
    text = PEAU.sub('', text)   # 👍🏻 et 👍 donnent le même mot
    return emoji.replace_emoji(text, replace=nom_emoji)


def substitute_url(text, url_replacement='URLEXPR'):
    """Remplace les URL par un jeton (reprise de substitute_url du TD5)."""
    return URL.sub(url_replacement, text)


def normaliser_typographie(text):
    """Remplace apostrophes et guillemets typographiques par leur forme simple."""
    return text.translate(TYPO)


def nettoyer(text, minuscules=True):
    """Enchaîne tous les nettoyages dans l'ordre.

    minuscules=False garde les majuscules : spaCy en a besoin pour repérer les prénoms (étape C).
    """
    text = remplacer_br(text)           # avant les URL : sinon "www.site.com<br/>" garde la balise
    if minuscules:
        text = mettre_en_minuscules(text)   # avant les jetons, qui restent en majuscules
    text = substitute_url(text)
    text = remplacer_emojis(text)
    text = normaliser_typographie(text)
    return normaliser_espaces(text)     # en dernier : nettoie les espaces ajoutés autour des jetons


def compter_doublons(df, col='comments_clean'):
    """Textes identiques présents dans plusieurs logements, du plus fréquent au moins fréquent."""
    par_texte = df.groupby(col)['listing_id'].agg(nb_avis='size', nb_logements='nunique')
    return par_texte[par_texte['nb_logements'] > 1].sort_values('nb_avis', ascending=False)
