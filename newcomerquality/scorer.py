import warnings
import mwapi
import datetime
import logging
from newcomerquality.config import load_mapper_model
from newcomerquality.sessionify import sessionify
import pandas as pd

from newcomerquality.make_features import make_features

logg = logging.getLogger()

class usercontribError(Exception):
    pass

class makeFeaturesError(Exception):
    pass

class registrationDateError(Exception):
    pass

#TODO transsform these functions into classmethods
def get_registration_date_of_user(user_id, context, mwapisession):
    '''get the registration timestamp of the user_id with context'''
    registration_date = None

    try:
        reg_data = mwapisession.get(action='query', list='users', usprop='registration', ususerids=user_id)
        registration_date_str = reg_data['query']['users'][0]['registration']
        registration_date = datetime.datetime.strptime(registration_date_str, "%Y-%m-%dT%H:%M:%SZ")
        logg.debug(f'registration date for {user_id} is {registration_date}')
        return registration_date
    except KeyError as e:
        raise registrationDateError

#TODO transsform these functions into classmethods
def get_edits_within_days_of_registration(user_id, context, registration_date, days, mwapisession):
    '''get the usercontribs'''
    end_date = (registration_date + datetime.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    # im as baffled as you as why `ucstart` is really what we want not `ucend`.
    user_q = mwapisession.get(action='query', list='usercontribs', ucuserids=user_id, ucstart=end_date)
    return user_q['query']['usercontribs']

#TODO make a class?
def score_newcomer_first_days(user_id, context='enwiki', days=2, registration_date=None, mwapisession=None):
    '''returns a list of goodfaith predicitons of newcomers sessions with first `days` of registering'''
    if not mwapisession:
        mwapisession = mwapi.Session("https://en.wikipedia.org",
                                     user_agent="Newcomerquality scoring client")

    # fn: use mediawiki to get the registration date
    if not registration_date: # it wasn't passed into the function
        registration_date = get_registration_date_of_user(user_id, context, mwapisession)
        if not registration_date: # this means that we could get it from the API
            return {'error': True, 'reason': 'could not get a registration date for user.'}
    else:
        #it was passed in, but we want to make sure its a datetime
        assert isinstance(registration_date, datetime.datetime)

    assert registration_date, 'We really need to have the registration date by now.'

    user_df = pd.DataFrame.from_dict({'user_id': [user_id], 'registration_date': [registration_date]})

    # fn: get all edits within days of registration date
    usercontribs = get_edits_within_days_of_registration(user_id, context, registration_date, days, mwapisession)

    if not usercontribs:
        raise usercontribError('could not get user contribs. maybe there were none or the they were deleted')
    # fn: session-revisions

    sessions_df = sessionify(user_df, usercontribs)

    sessions_featured_df = make_features(sessions_df, train_or_predict='predict')

    if len(sessions_featured_df) != len(sessions_df):
        raise makeFeaturesError('there was an error getting the making features')

    mapper, model = load_mapper_model()

    # print(sessions_featured_df)

    # fn: score_session(session=revision_list)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sessions_featured_mapped = mapper.transform(sessions_featured_df.copy())

    sessions_probas = model.predict_proba(sessions_featured_mapped)

    sessions_goodfaith_proba = sessions_probas[:, 1:]

    # question: how to aggregate multiple sessions (with multiple heuristics, and full underlying session scores).
    # return mulitiple session predictions

    newcomer_predictions = {
        'sessions_goodfaith_proba_mean': sessions_goodfaith_proba.mean(),
        'sessions_goodfaith_proba_min': sessions_goodfaith_proba.min(),
        'sessions_goodfaith_proba_max': sessions_goodfaith_proba.max(),
    }

    user_summary = {'days_of_revisions': days,
                    'edits_found': len(usercontribs),
                    'sessions_found': len(sessions_df),
                    }

    return_dict = {'error': False,
                   'user_id': user_id,
                   'user_summary': user_summary,
                   'scores': list(sessions_goodfaith_proba.flatten()),
                   'newcomer_predictions': newcomer_predictions, }

    return return_dict
