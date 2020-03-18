import os

from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample, EvaluatorHoldout
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from sys import exit

from scipy.sparse import csr_matrix

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


def print_popularity_statistics(matrix):
    popularity = get_popularity(matrix)
    print('=========================================')
    print('Item Popularity:\n', popularity)
    print('Max Popularity:', popularity.max())
    print('Sorted Item Popul.:\n', np.flip(np.sort(popularity)))
    print('Sorted Indexes of Item Popul.:\n', np.flip(np.argsort(popularity)))
    print('=========================================')


def print_matrix_statistics(matrix):
    users = matrix.shape[0]
    items = matrix.shape[1]
    preferences = np.sum(matrix.A)
    popularity = get_popularity(matrix)
    k = 0
    for i in range(len(popularity)):
        if popularity[i] == 0:
            k += 1
    print('=========================================')
    print('Number of Users:', users)
    print('Number of Items:', items)
    print('Number of Preferences:', preferences)
    print('AVG items per user:', preferences / users)
    print('AVG users per item:', preferences / items)
    print('Number of items with 0 popularity:', k)
    print('=========================================')


def print_user_statistics(matrix, user):
    print('=========================================')
    print('Info about user n.', user)
    print('Preferred items:', matrix[user])
    print('Number of preferred items:', np.sum(matrix[user]))
    print('=========================================')


def get_number_of_preferences(matrix):
    return np.sum(matrix.A)


# Loading Datasets
dataset = PinterestICCVReader()

URM_train = dataset.URM_train.copy()  # URM di training
URM_validation = dataset.URM_validation.copy()  # URM di validation
URM_test = dataset.URM_test.copy()  # URM di test
URM_test_negative = dataset.URM_test_negative.copy()
# print('negative:', URM_test_negative)

# TODO fondere test e training ed estrarre un nuovo test meno popolare


# Global positive info
URM = URM_train + URM_validation + URM_test
popularity = get_popularity(URM)

print('URM_train:')
print_matrix_statistics(URM_train)

print('URM_validation:')
print_matrix_statistics(URM_validation)

print('URM_test:')
print_matrix_statistics(URM_test)

print('URM_test_negative:')
print_matrix_statistics(URM_test_negative)

print('URM:')
print_matrix_statistics(URM)

exit('Stop for testing...')

number_of_users = URM.shape[0]
number_of_items = URM.shape[1]

verbose = False

URM_test_ = csr_matrix((number_of_users, number_of_items))
print('type:', type(URM_train))

for user in range(number_of_users):
    start = URM.indptr[user]
    stop = URM.indptr[user + 1]
    preferred_items = URM.indices[start:stop]
    number_of_preferred_items = len(preferred_items)

    popularity_of_preferred_items = [popularity[preferred_items[i]] for i in range(number_of_preferred_items)]
    less_popular_preferred_item = preferred_items[np.argmin(popularity_of_preferred_items)]

    # print('coo:',(user, less_popular_preferred_item))
    URM_test_[user, less_popular_preferred_item] = 1
    URM[user, less_popular_preferred_item] = 0

    if user % 1000 == 0:
        print('user', user)

    if verbose:
        print('User', user, '---------------------------------------')
        print('Preferred items:', preferred_items)
        print('Number of preferred items:', number_of_preferred_items)
        print('Popularity array:', popularity_of_preferred_items)
        print('Less popular item:', less_popular_preferred_item)

print('URM (updated):')
print_matrix_statistics(URM)

print('URM_test_:')
print_matrix_statistics(URM_test_)

cutoff = 5

CMN_wrapper_train = CMN_RecommenderWrapper(URM)
user_KNNCF_Recommender = UserKNNCFRecommender(URM)
item_KNNCF_Recommender = ItemKNNCFRecommender(URM)
rp3_beta_Recommender = RP3betaRecommender(URM)

evaluator_negative_item_sample = EvaluatorNegativeItemSample(URM_test_, URM_test_negative, cutoff_list=[5, 10])

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
