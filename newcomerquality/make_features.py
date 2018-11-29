from ores import api as ores_api

def make_features(df, feature_list):
    """

    :param df dataframe containig session-orient rows and a 'revids' column containg list of revids:
    :return: same dataframe with features added from feature_list
    """
    # find all the revids
    list_of_rev_ids = list(df['rev_ids'])
    all_revids = [y for x in list_of_rev_ids for y in x]  #flatten
    all_scores = get_ores_data_dgf(rev_ids=all_revids)
    revids_scores = dict(zip(all_revids, all_scores))
    from IPython import embed; embed()
    # make a map from revid to ores data


    # df['ores_data'] = df['revids'].apply(get_ores_data_from_map)


def get_ores_data_dgf(rev_ids, context='enwiki'):
    session = ores_api.Session(
        'https://ores.wikimedia.org',
        user_agent='newcomerquality',
        batch_size=50,
        parallel_requests=4,
        retries=2)

    scores = session.score(context, ('damaging', 'goodfaith'), rev_ids)
    return scores

