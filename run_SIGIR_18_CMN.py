#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""
from sklearn.preprocessing import MinMaxScaler

from settings import *
from settings import Settings, set_parameters
from Conferences.SIGIR.CMN_github.util.cmn import CollaborativeMemoryNetwork
from Conferences.SIGIR.CMN_github.util.gmf import PairwiseGMF
from Conferences.SIGIR.CMN_github.util.layers import LossLayer
from Recommender_import_list import *
from Conferences.SIGIR.CMN_our_interface.CMN_RecommenderWrapper import CMN_RecommenderWrapper

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative

import traceback, os, multiprocessing, pickle
from functools import partial
import numpy as np

from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderParameters

from Utils.print_results_latex_table import print_time_statistics_latex_table, print_results_latex_table, \
    print_parameters_latex_table
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('GPU:', tf.test.is_gpu_available())


def read_data_split_and_search_CMN(dataset_name):
    from Conferences.SIGIR.CMN_our_interface.CiteULike.CiteULikeReader import CiteULikeReader
    from Conferences.SIGIR.CMN_our_interface.Pinterest.PinterestICCVReader import PinterestICCVReader
    from Conferences.SIGIR.CMN_our_interface.Epinions.EpinionsReader import EpinionsReader

    if dataset_name == "citeulike":
        dataset = CiteULikeReader()

    elif dataset_name == "epinions":
        dataset = EpinionsReader()

    elif dataset_name == "pinterest":
        dataset = PinterestICCVReader()

    output_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)

    URM_train = dataset.URM_train.copy()
    URM_validation = dataset.URM_validation.copy()
    URM_test = dataset.URM_test.copy()
    URM_test_negative = dataset.URM_test_negative.copy()

    test_mode = False
    limit = False
    if limit:
        p = 700
        URM_train = URM_train[:p, :]
        URM_validation = URM_validation[:p, :]
        URM_test = URM_test[:p, :]
        URM_test_negative = URM_test_negative[:p, :]

        '''
        user: 3
        is_relevant_current_cutoff: [ True  True  True False False]
        recommended_items_current_cutoff: [  65   86   68 3671 1341]
        Warning! is_relevant_current_cutoff.sum()>1: 3
        relevant_items: [65 68 81 86]
        relevant_items_rating: [1. 1. 1. 1.]
        items_to_compute: 
        [  42   62   65   68   81   86  148  218  559  662  776  792 1164 1341
         1418 1491 1593 1603 1617 1697 2140 2251 2446 2517 2566 2643 2719 2769
         2771 3081 3133 3161 3188 3268 3409 3666 3671 3845 3864 3897 3984 4272
         4327 4329 4431 4519 4565 4568 4718 4812 4915 5096 5128 5137 5141 5184
         5217 5241 5371 5394 5415 5492 5521 5775 5798 5830 5831 5931 6005 6281
         6375 6558 6638 6644 6661 6705 6881 6898 6939 6970 7010 7018 7147 7224
         7327 7404 7453 7466 7475 7561 7764 8064 8102 8222 8368 8530 8957 9101
         9322 9368 9619 9782 9832]
        '''
        print('USER 3')

        print('test ', URM_test[3])
        print('train ', URM_train[3])
        print('valid ', URM_validation[3])
        print('neg ', URM_test_negative[3])

        # Durante l'esecuzione era stato notato un HR>1. Il motivo e' che veniva calcolato sul validation set (che per ogni utente ha
        # piu' oggetti preferiti (non uno)
        # Alla fine l'HR sara' minore o uguale ad uno perche' e' calcolato sul test set.

    popularity = get_popularity(URM_train)

    min_value = np.min(popularity)
    max_value = np.max(popularity)
    gap = max_value - min_value

    popularity = (popularity - min_value) / gap

    print('Luciano > min:', min_value)
    print('Luciano > max:', max_value)
    print('Luciano > normalized popularity:', popularity)

    set_parameters(
        popularity=popularity,
        loss_alpha=200,
        loss_beta=0.02,
        loss_scale=1,
        loss_percentile=get_percentile(popularity, 45),

        metrics_alpha=100,
        metrics_beta=0.03,
        metrics_gamma=5,
        metrics_scale=1 / 15,
        metrics_percentile=0.45,
        new_loss = False
    )

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    collaborative_algorithm_list = [
        Random,
        TopPop,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
    ]

    # metric_to_optimize = "WEIGHTED_HIT_RATE"
    # metric_to_optimize = "HIT_RATE"
    metric_to_optimize = "CUSTOM_HIT_RATE"

    print('metric_to_optimize:', metric_to_optimize)

    # Ensure IMPLICIT data and DISJOINT sets
    assert_implicit_data([URM_train, URM_validation, URM_test, URM_test_negative])

    if dataset_name == "citeulike":
        assert_disjoint_matrices([URM_train, URM_validation, URM_test])
        assert_disjoint_matrices([URM_test, URM_test_negative])

    elif dataset_name == "pinterest":
        assert_disjoint_matrices([URM_train, URM_validation, URM_test])
        assert_disjoint_matrices([URM_train, URM_validation, URM_test_negative])

    else:
        assert_disjoint_matrices([URM_train, URM_validation, URM_test, URM_test_negative])

    algorithm_dataset_string = "{}_{}_".format(ALGORITHM_NAME, dataset_name)

    plot_popularity_bias([URM_train + URM_validation, URM_test],
                         ["URM train", "URM test"],
                         output_folder_path + algorithm_dataset_string + "popularity_plot")

    save_popularity_statistics([URM_train + URM_validation, URM_test],
                               ["URM train", "URM test"],
                               output_folder_path + algorithm_dataset_string + "popularity_statistics")

    from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample

    evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=[5])
    if not test_mode:
        # evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=[5, 10])
        evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=[5])
    else:
        evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=[5])

    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train=URM_train,
                                                       metric_to_optimize=metric_to_optimize,
                                                       evaluator_validation_earlystopping=evaluator_validation,
                                                       evaluator_validation=evaluator_validation,
                                                       evaluator_test=evaluator_test,
                                                       output_folder_path=output_folder_path,
                                                       parallelizeKNN=False,
                                                       allow_weighting=True,
                                                       n_cases=35)

    # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    # resultList = pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)
    #
    # pool.close()
    # pool.join()

    for recommender_class in collaborative_algorithm_list:

        try:
            if not test_mode:
                runParameterSearch_Collaborative_partial(recommender_class)
            else:
                print('skipping', recommender_class)

        except Exception as e:

            print("On recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()

    ################################################################################################
    ###### CMN

    try:

        temp_file_folder = output_folder_path + "{}_log/".format(ALGORITHM_NAME)

        CMN_article_parameters = {
            "epochs": 100,
            "epochs_gmf": 100,
            "hops": 3,
            "neg_samples": 4,
            "reg_l2_cmn": 1e-1,
            "reg_l2_gmf": 1e-4,
            "pretrain": True,
            # "pretrain": False,
            "learning_rate": 1e-3,
            "verbose": False,
            "temp_file_folder": temp_file_folder
        }

        if dataset_name == "citeulike":
            CMN_article_parameters["batch_size"] = 128
            CMN_article_parameters["embed_size"] = 50

        elif dataset_name == "epinions":
            CMN_article_parameters["batch_size"] = 128
            CMN_article_parameters["embed_size"] = 40

        elif dataset_name == "pinterest":
            CMN_article_parameters["batch_size"] = 256
            CMN_article_parameters["embed_size"] = 50

        CMN_earlystopping_parameters = {
            "validation_every_n": 5,
            "stop_on_validation": True,
            "evaluator_object": evaluator_validation,
            "lower_validations_allowed": 5,
            "validation_metric": metric_to_optimize
        }

        parameterSearch = SearchSingleCase(CMN_RecommenderWrapper,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

        recommender_parameters = SearchInputRecommenderParameters(
            CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
            FIT_KEYWORD_ARGS=CMN_earlystopping_parameters)

        parameterSearch.search(recommender_parameters,
                               fit_parameters_values=CMN_article_parameters,
                               output_folder_path=output_folder_path,
                               output_file_name_root=CMN_RecommenderWrapper.RECOMMENDER_NAME)




    except Exception as e:

        print("On recommender {} Exception {}".format(CMN_RecommenderWrapper, str(e)))
        traceback.print_exc()

    n_validation_users = np.sum(np.ediff1d(URM_validation.indptr) >= 1)
    n_test_users = np.sum(np.ediff1d(URM_test.indptr) >= 1)

    print_time_statistics_latex_table(result_folder_path=output_folder_path,
                                      dataset_name=dataset_name,
                                      results_file_prefix_name=ALGORITHM_NAME,
                                      other_algorithm_list=[CMN_RecommenderWrapper],
                                      ICM_names_to_report_list=[],
                                      n_validation_users=n_validation_users,
                                      n_test_users=n_test_users,
                                      n_decimals=2)
    if not test_mode:
        print_results_latex_table(result_folder_path=output_folder_path,
                                  results_file_prefix_name=ALGORITHM_NAME,
                                  dataset_name=dataset_name,
                                  metrics_to_report_list=["HIT_RATE", "NDCG"],
                                  # cutoffs_to_report_list=[5, 10],
                                  cutoffs_to_report_list=[5],
                                  ICM_names_to_report_list=[],
                                  other_algorithm_list=[CMN_RecommenderWrapper])
    else:
        print_results_latex_table(result_folder_path=output_folder_path,
                                  results_file_prefix_name=ALGORITHM_NAME,
                                  dataset_name=dataset_name,
                                  metrics_to_report_list=["HIT_RATE", "NDCG"],
                                  cutoffs_to_report_list=[5],
                                  ICM_names_to_report_list=[],
                                  other_algorithm_list=[CMN_RecommenderWrapper])


if __name__ == '__main__':

    print('Luciano > Experiment started!')

    ALGORITHM_NAME = "CMN"
    CONFERENCE_NAME = "SIGIR"

    # dataset_list = ["citeulike", "pinterest", "epinions"]
    # dataset_list = ["citeulike"]
    dataset_list = ["pinterest"]
    # dataset_list = ["epinions"]

    for dataset in dataset_list:
        read_data_split_and_search_CMN(dataset)

    print_parameters_latex_table(result_folder_path="result_experiments/{}/".format(CONFERENCE_NAME),
                                 results_file_prefix_name=ALGORITHM_NAME,
                                 experiment_subfolder_list=dataset_list,
                                 ICM_names_to_report_list=[],
                                 other_algorithm_list=[CMN_RecommenderWrapper])

    print('Luciano > Done!')
