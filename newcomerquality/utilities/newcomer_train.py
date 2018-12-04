r"""
Trains newcomer model expecting dataframe input

Usage:
    newcomer_train -h | --help
    newcomer_train                  (--dump-file=<dumpf>)
                                    (--fn=<training_fn>)
                                    [--model_params=<model_params>]
                                    [--model_file=<modelf>]
                                    (--scaling_mapper=<scaling_mapper>)

Options:
    -h --help                   Prints out this documentation.
    --dump-file=<dumpf>         Path to dump file.
    --fn=<training_fn>          Part of training to run, either save_scaling_mapper export, tuning_report, or save_model
    --model_params=<model_params>   Path to look for yaml of model params.
    --scaling_mapper=<scaling_mapper> Path to look for the scaling mapper.
    --model_file=<modelf>       Path to dump the model file
"""
import json
import logging
import sys
import pickle
import docopt

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn_pandas import DataFrameMapper
import sklearn

logger = logging.getLogger(__name__)


def make_scaling_mapper(dumpf, scaling_mapper):
    df = pd.read_json(dumpf)

    mapper = DataFrameMapper([
        (['goodfaith_scores_mean'], sklearn.preprocessing.StandardScaler()),
        (['goodfaith_scores_var'], sklearn.preprocessing.StandardScaler()),
        (['goodfaith_scores_max'], sklearn.preprocessing.StandardScaler()),
        (['goodfaith_scores_min'], sklearn.preprocessing.StandardScaler()),
        (['goodfaith_scores_reg_slope'], sklearn.preprocessing.StandardScaler()),
        (['goodfaith_scores_reg_intercept'], sklearn.preprocessing.StandardScaler()),
        (['goodfaith_scores_count'], sklearn.preprocessing.StandardScaler()),
        (['goodfaith_scores_count_log'], sklearn.preprocessing.StandardScaler()),
        (['goodfaith_timestamps_total_seconds'], sklearn.preprocessing.StandardScaler()),
        (['goodfaith_timestamps_variance'], sklearn.preprocessing.StandardScaler()),
        (['goodfaith_timestamps_min'], sklearn.preprocessing.StandardScaler()),
        (['goodfaith_timestamps_max'], sklearn.preprocessing.StandardScaler()),
        (['self_reverts'], sklearn.preprocessing.StandardScaler()),
        (['edit_wars'], sklearn.preprocessing.StandardScaler()),
        (['pages_unique_count'], sklearn.preprocessing.StandardScaler()),
        (['pages_namespace_count'], sklearn.preprocessing.StandardScaler()),
        (['pages_nonmain_count'], sklearn.preprocessing.StandardScaler()),
        (['pages_talk_count'], sklearn.preprocessing.StandardScaler()),
        (['singleton_session'], sklearn.preprocessing.StandardScaler()),
        (['first_edit_ores_goodfaith'], sklearn.preprocessing.StandardScaler()),
        (['first_edit_ores_damaging'], sklearn.preprocessing.StandardScaler()),
        (['any_edit_ores_goodfaith'], sklearn.preprocessing.StandardScaler()),
        (['any_edit_ores_damaging'], sklearn.preprocessing.StandardScaler()),
    ])

    data = mapper.fit_transform(df.copy())
    pickle.dump(mapper, open(scaling_mapper, 'wb'))


def load_dataframe_as_arrays(dumpf, scaling_mapper):
    mapper = pickle.load(open(scaling_mapper, 'rb'))
    df = pd.read_json(dumpf)

    y = df['goodfaith_label'].apply(lambda x: int(x))
    data = mapper.transform(df.copy())
    X = data
    return y, X


def tuning_report(dumpf, scaling_mapper, model_params):
    ## Copied from Rayid Ghani's MagicLoops repo

    y, X = load_dataframe_as_arrays(dumpf, scaling_mapper)
    grid = json.load(open(model_params,'r'))

    def joint_sort_descending(l1, l2):
        # l1 and l2 have to be numpy arrays
        idx = np.argsort(l1)[::-1]
        return l1[idx], l2[idx]

    def generate_binary_at_k(y_scores, k):
        cutoff_index = int(len(y_scores) * (k / 100.0))
        test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
        return test_predictions_binary

    def precision_at_k(y_true, y_scores, k):
        # y_scores_sorted, y_true_sorted = zip(*sorted(zip(y_scores, y_true), reverse=True))
        y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
        preds_at_k = generate_binary_at_k(y_scores_sorted, k)
        # precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
        # precision = precision[1]  # only interested in precision for label 1
        precision = precision_score(y_true_sorted, preds_at_k)
        return precision


    def clf_loop(models_to_run, clfs, grid, X, y):
        """Runs the loop using models_to_run, clfs, gridm and the data
        """
        results_df = pd.DataFrame(columns=('model_type', 'clf', 'parameters', 'p_at_20', 'p_at_40'))
        for n in range(1, 2):
            # create training and valdation sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)
            for index, clf in enumerate([clfs[x] for x in models_to_run]):
                print(models_to_run[index])
                parameter_values = grid[models_to_run[index]]
                for p in ParameterGrid(parameter_values):
                    try:
                        clf.set_params(**p)
                        y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
                        # you can also store the model, feature importances, and prediction scores
                        # we're only storing the metrics for now
                        y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                        results_df.loc[len(results_df)] = [models_to_run[index], clf, p,
                                                           precision_at_k(y_test_sorted, y_pred_probs_sorted, 20.0),
                                                           precision_at_k(y_test_sorted, y_pred_probs_sorted, 40.0)]
                    except IndexError as e:
                        print('Error:', e)
                        continue
        return results_df

    models_to_run = ['GB', 'LR']

    results_df = clf_loop(models_to_run, {'LR': LogisticRegression(), 'GB': GradientBoostingClassifier()}, grid, X, y)
    results_df.sort_values('p_at_40', ascending=False).to_html(sys.stdout)

def create_model(dumpf, scaling_mapper, model_params, modelf):
    y, X = load_dataframe_as_arrays(dumpf, scaling_mapper)
    clf_defaults = json.load(open(model_params,'r'))
    clf_dict = {'LR': LogisticRegression(), 'GB': GradientBoostingClassifier()}
    clf_to_use = list(clf_defaults.keys())[0]
    clf = clf_dict[clf_to_use]
    clf.set_params(**clf_defaults[clf_to_use])

    fitted_model = clf.fit(X, y)

    pickle.dump(fitted_model, open(modelf, 'wb'))


def main(argv=None):
    args = docopt.docopt(__doc__, argv=argv)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(name)s -- %(message)s'
    )

    dumpf = args['--dump-file']
    training_fn = args['--fn']
    scaling_mapper = args['--scaling_mapper']
    model_params = args['--model_params']
    modelf = args['--model_file']

    if training_fn == 'make_scaling_mapper':
        make_scaling_mapper(dumpf, scaling_mapper)
    elif training_fn == 'tuning_report':
        tuning_report(dumpf, scaling_mapper, model_params)
    elif training_fn == 'create_model':
        create_model(dumpf, scaling_mapper, model_params, modelf)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("\n^C Caught.  Exiting...")
