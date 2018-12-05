import mwapi
import datetime
import logging
from sessionify import sessionify
import pandas as pd

from make_features import make_features


def get_registration_date_of_user(user_id, context, mwapisession):
    '''get the registration timestamp of the user_id with context'''
    registration_date = None

    reg_data = mwapisession.get(action='query', list='users', usprop='registration', ususerids=user_id)
    registration_date_str = reg_data['query']['users'][0]['registration']
    registration_date = datetime.datetime.strptime(registration_date_str, "%Y-%m-%dT%H:%M:%SZ")
    logg.debug(f'registration date for {user_id} is {registration_date}')
    return registration_date


def get_edits_within_days_of_registration(user_id, context, registration_date, days, mwapisession):
    '''get the usercontribs'''
    end_date = (registration_date + datetime.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    #im as baffled as you as why `ucstart` is really what we want not `ucend`.
    user_q = mwapisession.get(action='query', list='usercontribs', ucuserids=user_id, ucstart=end_date)
    return user_q['query']['usercontribs']


def score_newcomer_first_days(user_id, context='enwiki', days=2, registration_date=None, mwapisession=None):
    '''returns a list of goodfaith predicitons of newcomers sessions with first `days` of registering'''
    if not mwapisession:
        mwapisession = mwapi.Session("https://en.wikipedia.org",
                        user_agent="Newcomerquality scoring client")

    # fn: use mediawiki to get the registration date
    if not registration_date:
        registration_date = get_registration_date_of_user(user_id, context, mwapisession)
    else:
        assert isinstance(registration_date, datetime.datetime)
    assert registration_date, 'this means that we could get it from the API'
    user_df = pd.DataFrame.from_dict({'user_id':[user_id], 'registration_date':[registration_date]})

    # fn: get all edits within days of registration date
    usercontribs = get_edits_within_days_of_registration(user_id, context, registration_date, days, mwapisession)

    if not usercontribs:
        return None
    # fn: session-revisions

    session_df = sessionify(user_df, usercontribs)

    session_featured_df = make_features(session_df, train_or_predict='predict')

    print(session_featured_df)
    # fn: score_session(session=revision_list)

    # fn: score multiple sessions with model (in loop?)
    # question: how to aggregate multiple sessions (with multiple heuristics, and full underlying session scores).
    # return mulitiple session predictions



if __name__ == '__main__':

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s:%(name)s -- %(message)s',
        filename='logs/newcomer_prediction.log', filemode='w',
    )
    logg = logging.getLogger()

    test_user_ids = (44846, 4670490, 35303278, 11801436, 1755837)
    for test_user_id in test_user_ids:
        user_score = score_newcomer_first_days(test_user_id)
        if not user_score:
            logg.info(f'For some reason could not get contrib data for user {test_user_id}')
        else:
            logg.info(f'User score for user {test_user_id} is {user_score}')
