import pandas as pd
import datetime as dt

def make_sessions(rev_df):
    # these structures store the timestamps
    rev_df = rev_df.sort_values('timestamp')
    edit_sessions = []
    curr_edit_session = []

    # initialize prev to the earliest data possible
    prev_timestamp = dt.datetime(year=2001, month=1, day=1)

    for index, (globindex, row) in enumerate(rev_df.iterrows()):
        #         print('index:', index)
        curr_timestamp = dt.datetime.strptime(row['timestamp'], "%Y-%m-%dT%H:%M:%SZ")
        revid = row['revid']
        # if curr timestamp within 1 hour of last then append
        if curr_timestamp - prev_timestamp < dt.timedelta(hours=1):
            curr_edit_session.append({'timestamp': curr_timestamp, 'revid': revid})
        # else start a new edit session
        else:
            # if there's a pre-existing session save it to the return
            if curr_edit_session:
                edit_sessions.append(curr_edit_session)
            # and start a new session
            curr_edit_session = [{'timestamp': curr_timestamp, 'revid': revid}]
        # this is before
        if index < len(rev_df) - 1:
            prev_timestamp = curr_timestamp
        # this is the last item save this session too.
        else:
            #             print('this is the end')
            edit_sessions.append(curr_edit_session)

    return edit_sessions


def sessionify(user_df, usercontribs):
    """

    :param usercontribs, usuall returned from MediaWiki via mwapi:
    :return: session dataframe
    """

    user_session_dfs = []
    rev_df = pd.DataFrame.from_dict(usercontribs)
    #     print(rev_df)
    sessions = make_sessions(rev_df)
    print(f'found {len(sessions)} sessions for {len(rev_df)} revisions')
    #     print(user_df['user_id'].iloc[0])

    for edit_session in sessions:
        session_df = user_df.copy()
        # let me explain this next line, you can't assign a list to cell in a standard way because
        # pandas will try and line it as if you are adding a column, so do some gymnastics
        sess_revids = [e['revid'] for e in edit_session]
        sess_timestamps = [e['timestamp'] for e in edit_session]
        sess_contribs = [c for c in usercontribs if c['revid'] in sess_revids]
        sess_page_dicts = [{'page_id': c['pageid'], 'page_ns': c['ns']} for c in sess_contribs]

        # from IPython import embed; embed()
        session_df['rev_ids'] = None
        session_df['timestamps'] = None
        session_df['pages'] = None

        session_df.at[0, 'rev_ids'] = sess_revids
        session_df.at[0, 'timestamps'] = sess_timestamps
        session_df.at[0, 'pages'] = sess_page_dicts

        user_session_dfs.append(session_df)

    return pd.concat(user_session_dfs)

