import mwapi


def score_newcomer_first_days(user_id, context='enwiki', days=2):
    '''returns a list of goodfaith predicitons of newcomers sessions with first `days` of registering'''
    # fn: use mediawiki to get the registration date
    # fn: get all edits within days of registration date
    # fn: session-revisions
    # fn: score_session(session=revision_list)
    # fn: get ores scores revisions
    # fn: score multiple sessions with model (in loop?)
    # question: how to aggregate multiple sessions (with multiple heuristics, and full underlying session scores).
    # return mulitiple session predictions



if __name__ == '__main__':
