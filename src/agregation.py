"""Étape D : un document par logement (la table finale utilisée par les modèles)."""

# les versions du texte produites à l'étape C (texte nettoyé + 4 versions prétraitées)
VERSIONS = ['comments_clean', 'text_lemmes', 'text_stop', 'text_neg', 'text_pos']


def agreger_par_logement(df, min_avis=10):
    """Une ligne par logement : ses avis mis bout à bout (pour chaque version du texte) et ses notes."""
    nb_avis = df.groupby('listing_id').size()
    df = df[df['listing_id'].isin(nb_avis[nb_avis >= min_avis].index)]

    textes = df.groupby('listing_id')[VERSIONS].agg(' '.join)
    # les notes sont les mêmes pour tous les avis d'un logement : on prend la première
    notes = [c for c in df.columns if c.startswith('review_scores')]
    infos = df.groupby('listing_id')[['quartier', 'type_logement'] + notes].first()

    logements = textes.join(infos)
    logements.insert(0, 'nb_avis', df.groupby('listing_id').size())
    return logements
