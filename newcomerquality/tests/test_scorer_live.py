import datetime
from newcomerquality.scorer import get_registration_date_of_user, score_newcomer_first_days
import logging
from unittest.mock import patch, MagicMock

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(name)s -- %(message)s',
    filename='logs/newcomer_prediction.log', filemode='w',
)
logg = logging.getLogger()


def test_get_registration_date_of_user():
    user_id = 123
    context = 'enwiki'
    mwapisession = MagicMock()
    mwapisession.get.return_value = {
        'query': {
            'users': [
                {
                    'registration': '2018-04-01T00:01:02Z',
                }
            ]
        }
    }
    registration_date = get_registration_date_of_user(user_id, context, mwapisession)
    assert registration_date == datetime.datetime(2018, 4, 1, 0, 1, 2)


def test_scorer_live():
    '''needs an active connection'''
    for test_user_id in test_user_ids:
        user_score_ret = score_newcomer_first_days(test_user_id)
        print('predict user runtime took {t1-t0}')
        if user_score_ret['error']:
            logg.info(f'For some reason could not get contrib data for user {test_user_id}')
        elif not user_score_ret['error']:
            logg.info(f"User score for user {test_user_id} is {user_score_ret['user_id']}")
            print(user_score_ret['newcomer_predictions']['sessions_goodfaith_proba_min'])
