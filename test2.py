import os

from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample, EvaluatorHoldout
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

print('Tf Version:', tf.__version__)
print('GPU:', tf.test.is_gpu_available())
# print("Num GPUs Available: ", len(tf.experimental.list_physical_devices('GPU')))

import numpy as np

from Conferences.SIGIR.CMN_our_interface.CMN_RecommenderWrapper import CMN_RecommenderWrapper
from Conferences.SIGIR.CMN_our_interface.Pinterest.PinterestICCVReader import PinterestICCVReader


def get_popularity(matrix):
    return matrix.A.sum(axis=0)


def print_statistics(matrix):
    popularity = get_popularity(matrix)
    print('=========================================')
    print('Number of Items:\n', len(matrix.A[0]))
    print('Item Popularity:\n', popularity)
    print('Max Popularity:\n', popularity.max())
    print('Sorted Item Popul.:\n', np.flip(np.sort(popularity)))
    print('Sorted Indexes of Item Popul.:\n', np.flip(np.argsort(popularity)))


def test(matrix, user):
    print('=========================================')
    print('Info about user n.', user)
    print('Preferred items:', matrix[user])
    print('Number of preferred items:', np.sum(matrix[user]))
    print('=========================================')


# Loading Datasets
dataset = PinterestICCVReader()

URM_train = dataset.URM_train.copy()  # URM di training
URM_validation = dataset.URM_validation.copy()  # URM di validation
URM_test = dataset.URM_test.copy()  # URM di test
URM_test_negative = dataset.URM_test_negative.copy()
# print('negative:', URM_test_negative)

# TODO fondere test e training ed estrarre un nuovo test meno popolare.

URM = URM_train + URM_validation

popularity = get_popularity(URM)

number_of_users = URM_train.shape[0]
number_of_items = URM_train.shape[1]

print('URM_train.shape:', URM_train.shape)
print('URM_test.shape:', URM_test.shape)
print('URM_validation.shape:', URM_validation.shape)

print('number_of_users:', number_of_users)
print('number_of_items:', number_of_items)

# test(URM_train, 267)

cutoff = 5

print_statistics(URM_train)

CMN_wrapper_train = CMN_RecommenderWrapper(URM_train)
user_KNNCF_Recommender = UserKNNCFRecommender(URM_train)
item_KNNCF_Recommender = ItemKNNCFRecommender(URM_train)
rp3_beta_Recommender = RP3betaRecommender(URM_train)

evaluator_negative_item_sample = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=[5, 10])

CMN_wrapper_train.loadModel('result_experiments/SIGIR/CMN_pinterest/', 'CMN_RecommenderWrapper_best_model')
d, s = evaluator_negative_item_sample.evaluateRecommender(CMN_wrapper_train)
print('CMN_wrapper_train')
print(s)

user_KNNCF_Recommender.loadModel('result_experiments/SIGIR/CMN_pinterest/', 'UserKNNCFRecommender_cosine_best_model')
d, s = evaluator_negative_item_sample.evaluateRecommender(user_KNNCF_Recommender)
print('user_KNNCF_Recommender')
print(s)

item_KNNCF_Recommender.loadModel('result_experiments/SIGIR/CMN_pinterest/', 'ItemKNNCFRecommender_cosine_best_model')
d, s = evaluator_negative_item_sample.evaluateRecommender(item_KNNCF_Recommender)
print('item_KNNCF_Recommender')
print(s)

rp3_beta_Recommender.loadModel('result_experiments/SIGIR/CMN_pinterest/', 'RP3betaRecommender_best_model')
d, s = evaluator_negative_item_sample.evaluateRecommender(rp3_beta_Recommender)
print('rp3_beta_Recommender')
print(s)
