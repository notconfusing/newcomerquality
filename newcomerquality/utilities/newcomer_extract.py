r"""
Gathers the reverted status for a set of revisions and
prints a TSV to stdout of the format:

<rev_id>\t<reverted>\t<reason>

Usage:
    extract_damaging -h | --help
    extract_damaging                [--dump-file=<dumpf>]
                                    [--host=<url>]
                                    [--start=<date>]
                                    [--end=<date>]
                                    [--trusted-groups=<groups>]
                                    [--trusted-edits=<num>]
                                    [--revert-radius=<revs>]
                                    [--revert-window=<hrs>]
                                    [--reverted-only]
                                    [--check-blocked]
                                    [--verbose]
                                    [--rev-reverteds=<path>]

Options:
    -h --help                   Prints out this documentation.
    --dump-file=<dumpf>         Path to dump file.
    --host=<url>                The host URL of the MediaWiki install where an
                                API can be found.
    --start=<timestamp>         Start time.
    --end=<timestamp>           End time.
    --reverted-only             Only mark reverted edits as potentially
                                damaging
    --revert-radius=<revs>      The maximum amount of revisions that a
                                reverting edit can revert [default: 15]
    --revert-window=<hrs>       The maximum amount of time to wait for a
                                revision to be reverted [default: 48]
    --trusted-groups=<groups>   User groups that should be considered trusted.
                                Split by ",".
    --trusted-edits=<num>       Minimum number of edits to be considered
                                trusted.
    --check-blocked             Check if users are blocked.
    --verbose                   Prints dots and stuff to stderr
    --rev-reverteds=<path>      The location to write output to.
                                [default: <stdout>]
"""
import json
import logging
import sys
from collections import deque, namedtuple
from functools import lru_cache

import docopt
import mwapi
import mwreverts
import mwtypes

from mwtypes import Timestamp

import pandas as pd

from ..make_features import make_features

Revision = namedtuple("Revision", ['id', 'status', 'reason'])
User = namedtuple("User", ['id', 'editcount', 'groups'])

logger = logging.getLogger(__name__)


def main(argv=None):
    args = docopt.docopt(__doc__, argv=argv)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(name)s -- %(message)s'
    )

    revert_radius = int(args['--revert-radius'])
    revert_window = int(args['--revert-window']) * (60 * 60)  # secs --> hrs

    if args['--host']:
        session = mwapi.Session(args['--host'],
                                user_agent="ORES revert labeling utility")
    else:
        session = None
    dumpf = args['--dump-file']

    verbose = args['--verbose']
    start = args['--start']
    if start:
        start = Timestamp(start)
    end = args['--end']
    if end:
        end = Timestamp(end)
    reverted_only = args['--reverted-only']
    trusted_groups = args['--trusted-groups']
    if trusted_groups:
        trusted_groups = trusted_groups.split(',')
        trusted_users = load_user_group_members(trusted_groups, session)
    else:
        trusted_users = None
    trusted_edits = args['--trusted-edits']
    if trusted_edits:
        trusted_edits = int(trusted_edits)


    check_blocked = args['--check-blocked']
    run(dumpf, session, start, end, revert_radius, revert_window,
        reverted_only, trusted_users, trusted_edits,
        check_blocked, verbose=verbose)

def run(dumpf, session, start, end, revert_radius, revert_window,
        reverted_only, trusted_users, trusted_edits,
        check_blocked, verbose=False):
    completed_row_oriented = json.load(open(dumpf,'r'))
    df = pd.DataFrame.from_dict(completed_row_oriented)
    logger.info('Total rows from wikilabels: {}'.format(len(df)))
    df = df[pd.notnull(df['goodfaith_label'])]
    logger.info('Total rows with a non-skipped label: {}'.format(len(df)))
    # df = df.iloc[:50]
    logger.info('Total rows after subsetting for quick debugging: {}'.format(len(df)))
    featured = make_features(df, train_or_predict='train')
    featured.to_json(sys.stdout)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("\n^C Caught.  Exiting...")
