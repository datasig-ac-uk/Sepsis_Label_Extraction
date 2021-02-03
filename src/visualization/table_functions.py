import sys

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve, auc

sys.path.insert(0, '../')
import features.sepsis_mimic3_myfunction as mimic3_myfunc
from visualization.sepsis_mimic3_myfunction_patientlevel import output_at_metric_level

import constants

headers = ['H1', 'H2', 'H3']


def instance_level_auc_pd_threemodels(labels_list_list, probs_list_list, \
                                      models=constants.MODELS, definitions=constants.FEATURES, \
                                      pd_save_name=None):
    """
        instance level auc pd outout for all three models
    """
    results = []

    for model in range(len(models)):

        aucs = []

        for defi in range(len(definitions)):
            fpr, tpr, _ = roc_curve(labels_list_list[model][defi], probs_list_list[model][defi])
            roc_auc = auc(fpr, tpr)

            aucs.append(roc_auc)

        results.append([models[model], "{:.3f}".format(aucs[0]), \
                        "{:.3f}".format(aucs[1]), \
                        "{:.3f}".format(aucs[-1])])

    output_df = pd.DataFrame(results, columns=['Model', 'H1', 'H2', 'H3'])

    if pd_save_name is not None:

        output_df.to_csv(pd_save_name + ".csv")

    else:

        return output_df


def patient_level_auc_pd_threemodels(fprs_list_list, tprs_list_list, \
                                     models=constants.MODELS, headers=headers, \
                                     definitions=constants.FEATURES, \
                                     pd_save_name=None, \
                                     numerics_format="{:.3f}", for_write=True):
    """
        patient level auc pd output for all three models
    """
    results = []

    for model in range(len(models)):

        aucs = []

        for defi in range(len(definitions)):

            roc_auc = auc(fprs_list_list[model][defi], tprs_list_list[model][defi])

            if for_write:
                aucs += [numerics_format.format(roc_auc) + ' &']
            else:
                aucs += [numerics_format.format(roc_auc)]

        results.append([models[model]] + aucs)

    output_df = pd.DataFrame(results, columns=['Model'] + headers)

    if pd_save_name is not None:

        output_df.to_csv(pd_save_name + ".csv")

    else:

        return output_df


def patient_level_output_pd_threemodels(some_list_list, metric_seq_list_list, \
                                        models=constants.MODELS, headers=headers, definitions=constants.FEATURES, \
                                        metric_required=[0.375], numerics_format="{:.2%}", \
                                        operator=lambda x: x, for_write=True, pd_save_name=None):
    """
        patient level pd-format output for all three models
    """
    results = []

    for model in range(len(models)):

        outputs_current = []

        for defi in range(len(definitions)):

            output_current = output_at_metric_level(operator(some_list_list[model][defi]), \
                                                    metric_seq_list_list[model][defi], \
                                                    metric_required=metric_required)
            if for_write:
                outputs_current += [numerics_format.format(output_current) + ' &']
            else:
                outputs_current += [numerics_format.format(output_current)]

        results.append([models[model]] + outputs_current)

    output_df = pd.DataFrame(results, columns=['Model'] + headers)

    if pd_save_name is not None:

        output_df.to_csv(pd_save_name + ".csv")

    else:

        return output_df

