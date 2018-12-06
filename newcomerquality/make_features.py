import warnings

from ores import api as ores_api
import mwreverts.api
import mwapi
import datetime as dt
from collections import defaultdict
import numpy as np
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(name)s -- %(message)s',
    filename='logs/fetch.log', filemode='w',
)
logg = logging.getLogger()

session = mwapi.Session("https://en.wikipedia.org",
                        user_agent="Newcomer Quality Agent <max@notconfusing.com>")



def get_ores_data_dgf_from_api(rev_ids, context='enwiki'):
    session = ores_api.Session(
        'https://ores.wikimedia.org',
        user_agent='newcomerquality',
        batch_size=50,
        parallel_requests=4,
        retries=2)

    scores = session.score(context, ('damaging', 'goodfaith'), rev_ids)
    return scores


def get_ores_data_dgf(rev_ids, ores_data):
    ores_return = []
    for rev_id in rev_ids:
        try:
            ores_datum = ores_data[rev_id]
            ores_return.append(ores_datum)
        except KeyError:
            logg.info('It looks like we didnt pre-download ores data for rev_id:{rev_id}'.format(rev_id=rev_id))
    return ores_return


def num_reverts(revids):
    self_reverts = 0
    reverting_users = defaultdict(int)
    for rev_id in revids:
        try:
            _, reverted, reverted_to = mwreverts.api.check(
                session, rev_id, radius=2,  # most reverts within 5 edits
                window=48 * 60 * 60,  # 2 days
                rvprop={'user', 'ids'})  # Some properties we'll make use of
        except (RuntimeError, KeyError) as e:
            sys.stderr.write(str(e))
            continue

        if reverted is not None:

            reverting_user = reverted.reverting['user']
            reverting_users[reverting_user] += 1

            reverted_doc = [r for r in reverted.reverteds
                            if r['revid'] == rev_id][0]

            if 'user' not in reverted_doc or 'user' not in reverted.reverting:
                continue

            # self-reverts
            self_revert = \
                reverted_doc['user'] == reverting_user

            if self_revert:
                self_reverts += 1

    edit_war_users = [user for user, num_reverts in reverting_users.items() if num_reverts > 1]
    edit_wars = len(edit_war_users)

    return {'self_reverts': self_reverts, 'edit_wars': edit_wars}


def simp_lin_reg(vec, slope_intercept='slope'):
    if len(vec) == 1:
        return 0
    else:
        x = list(range(len(vec)))
        reg = np.polyfit(x, vec, 1)
        return reg[0] if slope_intercept == 'slope' else reg[1]


def total_seconds(timestamps):
    if len(timestamps) == 1:
        return 60 * 60  # one hour
    else:
        delta = max(timestamps) - min(timestamps)
        return 60 * 60 + delta.seconds  # one hour plus the difference


def fn_seconds(timestamps, fn):
    timestamps = sorted(timestamps)
    if len(timestamps) == 1:
        return 0  # no variance
    else:
        deltas = []
        for i in range(len(timestamps) - 1):
            deltas.append(timestamps[i + 1] - timestamps[1])
        delta_seconds = [d.seconds for d in deltas]
        return fn(delta_seconds)


def get_rev_timestamps(revids):
    """
    this method for when usercrontribs weren't prefetched
    :param revids:
    :return:
    """
    timestamps = []
    pages = []
    if len(revids) > 50:
        logg.info('truncating to 50 revids')
        revids = revids[:49]
    rev_query = session.get(action='query', prop='revisions', revids=revids)
    for page_id, page_info in rev_query['query']['pages'].items():
        page = {'page_id': page_id, 'page_ns': page_info['ns']}
        pages.append(page)
        revisions = page_info['revisions']
        for revision in revisions:
            # print(revision)
            timestamp = dt.datetime.strptime(revision['timestamp'], "%Y-%m-%dT%H:%M:%SZ")
            timestamps.append(timestamp)
    return {'timestamps': timestamps, 'pages': pages}


def get_last_user_contribs_upto_date(user_id, end_date):
    user_q = session.get(action='query', list='usercontribs', ucuserids=user_id, ucstart=end_date)
    return user_q['query']['usercontribs']


# def page_dicts_of_session(session_data):
#     sess_revids = [r['revid'] for r in session_data]
#     sess_contribs = [c for c in session_data['usercontribs'] if c['revid'] in sess_revids]
#     page_dicts = [{'page_id': c['pageid'], 'page_ns': c['ns']} for c in sess_contribs]
#     return page_dicts


def make_features(df, train_or_predict):
    """

    :param df dataframe containig session-orient rows and a 'revids' column containg list of revids:
    :return: same dataframe with features added from feature_list
    """
    ## Pre-cache some data from APIs in order to be able to do batching.
    # find all the revids
    list_of_rev_ids = list(df['rev_ids'])
    all_revids = [y for x in list_of_rev_ids for y in x]  # flatten
    all_scores = get_ores_data_dgf_from_api(rev_ids=all_revids)
    # make a map from revid to ores data
    revids_scores = dict(zip(all_revids, all_scores))

    assert train_or_predict in ('train', 'predict')

    if train_or_predict == 'train':
        df['timestamps_pages'] = df['rev_ids'].apply(lambda x: get_rev_timestamps(x))
        df['timestamps'] = df['timestamps_pages'].apply(lambda d: d['timestamps'])
        df['pages'] = df['timestamps_pages'].apply(lambda d: d['pages'])

    elif train_or_predict == 'predict':
        assert 'rev_ids' in df.columns
        assert 'timestamps' in df.columns
        assert 'pages' in df.columns

    df['ores_data'] = df['rev_ids'].apply(lambda rev_ids: get_ores_data_dgf(rev_ids, revids_scores))

    # check for empty ores or revert data
    def get_ores_scores(scores, damaging_goodfaith, probability_prediction):
        return_scores = []
        for score in scores:
            if isinstance(score, dict):
                try:
                    if probability_prediction == 'probability':
                        return_score = score[damaging_goodfaith]['score']['probability']['true']
                    elif probability_prediction == 'prediction':
                        return_score = score[damaging_goodfaith]['score']['prediction']

                    return_scores.append(return_score)
                except KeyError:
                    pass #i've seen this before if there was a textdeleted error
        return return_scores

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df['damaging_scores'] = df['ores_data'].apply(lambda scores: get_ores_scores(scores, 'damaging', 'probability'))
        df['damaging_predictions'] = df['ores_data'].apply(lambda scores: get_ores_scores(scores, 'damaging', 'prediction'))
        df['goodfaith_scores'] = df['ores_data'].apply(lambda scores: get_ores_scores(scores, 'damaging', 'probability'))
        df['goodfaith_predictions'] = df['ores_data'].apply(lambda scores: get_ores_scores(scores, 'damaging', 'prediction'))

        # missing_goodfaith = df[df['goodfaith_scores'].apply(lambda l: len(l) == 0)]
        # print('missing goodfaith were: {}'.format(len(missing_goodfaith)))

        df = df[df['goodfaith_scores'].apply(lambda l: len(l) > 0)]

        logg.info('getting revert data')
        df['revert_data'] = df['rev_ids'].apply(lambda x: num_reverts(x))


        df['self_reverts'] = df['revert_data'].apply(lambda d: d['self_reverts'])
        df['edit_wars'] = df['revert_data'].apply(lambda d: d['edit_wars'])
        # print('doing meta statistics')
        df['goodfaith_scores_mean'] = df['goodfaith_scores'].apply(np.mean)
        df['goodfaith_scores_var'] = df['goodfaith_scores'].apply(np.var)
        df['goodfaith_scores_max'] = df['goodfaith_scores'].apply(max)
        df['goodfaith_scores_min'] = df['goodfaith_scores'].apply(min)
        df['goodfaith_scores_reg_slope'] = df['goodfaith_scores'].apply(lambda v: simp_lin_reg(v, 'slope'))
        df['goodfaith_scores_reg_intercept'] = df['goodfaith_scores'].apply(lambda v: simp_lin_reg(v, 'intercept'))
        df['goodfaith_scores_count'] = df['goodfaith_scores'].apply(len)
        df['goodfaith_scores_count_log'] = df['goodfaith_scores'].apply(lambda v: np.log(len(v)))

        # print('doing time stats')

        df['goodfaith_timestamps_total_seconds'] = df['timestamps'].apply(total_seconds)
        df['goodfaith_timestamps_variance'] = df['timestamps'].apply(lambda t: fn_seconds(t, np.var))
        df['goodfaith_timestamps_min'] = df['timestamps'].apply(lambda t: fn_seconds(t, np.min))
        df['goodfaith_timestamps_max'] = df['timestamps'].apply(lambda t: fn_seconds(t, np.max))

        df['pages_unique_count'] = df['pages'].apply(lambda plist: len(set([p['page_id'] for p in plist])))
        df['pages_namespace_count'] = df['pages'].apply(lambda plist: len(set([p['page_ns'] for p in plist])))
        df['pages_nonmain_count'] = df['pages'].apply(
            lambda plist: len(set([p['page_ns'] for p in plist if p['page_ns'] != 0])))
        df['pages_talk_count'] = df['pages'].apply(
            lambda plist: len(set([p['page_ns'] for p in plist if p['page_ns'] % 2 == 1])))

        df['singleton_session'] = df['rev_ids'].apply(lambda rev_ids: len(rev_ids) == 1)

        df['first_edit_ores_goodfaith'] = df['goodfaith_predictions'].apply(lambda predictions: predictions[0])
        df['first_edit_ores_damaging'] = df['damaging_predictions'].apply(lambda predictions: predictions[0])
        df['any_edit_ores_goodfaith'] = df['goodfaith_predictions'].apply(lambda predictions: any(predictions))
        df['any_edit_ores_damaging'] = df['damaging_predictions'].apply(lambda predictions: any(predictions))
    return df
