import os

from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample

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

URM = URM_train + URM_validation

popularity = get_popularity(URM)

number_of_users = URM_train.shape[0]
number_of_items = URM_train.shape[1]

print('URM_train.shape:', URM_train.shape)
print('URM_test.shape:', URM_test.shape)
print('URM_validation.shape:', URM_validation.shape)

print('number_of_users:', number_of_users)
print('number_of_items:', number_of_items)

test(URM_train, 267)

print_statistics(URM)

CMN_wrapper_train = CMN_RecommenderWrapper(URM_train)
# CMN_wrapper_test = CMN_RecommenderWrapper(URM_test)

# Il modello che stiamo caricando e' stato addestrato su URM_train
# Ora al wrapper stiamo passando URM_test allo scopo di eseguire previsioni sul dataset di test

CMN_wrapper_train.loadModel('result_experiments/SIGIR/CMN_pinterest/', 'CMN_RecommenderWrapper_best_model')
# CMN_wrapper_test.loadModel('result_experiments/SIGIR/CMN_pinterest/', 'CMN_RecommenderWrapper_best_model')

# Il modello incluso nel wrapper non servira' direttamente. Il wrapper e' stato modificato in modo che la struttura del modello sia
# visualizzabile in tensorboard:
#
# tensorboard --logdir result_experiments/SIGIR/CMN_pinterest/CMN_log

# CMN_model = CMN_wrapper_train.model

'''
Il metodo che utilizziamo per le previsioni e' 

def _compute_item_score(self, user_id_array, items_to_compute=None):
    """

    :param user_id_array:       array containing the user indices whose recommendations need to be computed
    :param items_to_compute:    array containing the items whose scores are to be computed.
                                    If None, all items are computed, otherwise discarded items will have as score -np.inf
    :return:                    array (len(user_id_array), n_items) with the score.
    """
'''

cutoff = 5

evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=[5])

d, s = evaluator_test.evaluateRecommender(CMN_wrapper_train)
print(s)
print('STOP')

'''
users = [19, 20]
resources = [19, 10, 77]

result = CMN_wrapper._compute_item_score([1], resources)
# result = CMN_wrapper._compute_item_score(users)

recommended_items = CMN_wrapper.recommend(users, cutoff)
popularity_of_recomended_items = popularity

print('Scores:', result)

for i in range(len(users)):
    print('----------------------------------------------')
    print('User {} (cutoff {}):'.format(users[i], cutoff))
    print('Recommended items:', recommended_items[i])
    print('Popularity of recommended items:', popularity[recommended_items[i]])

'''

''' 
La procedura di valutazione di HR@n e' la seguente:

Per ogni utente si dividono gli oggetti che preferisce in due insiemi: 

    1) il primo e' incluso nel training set (80%) 
    2) il second e' incluso nel test set (20%)

Il sistema si addestra sul training set.
A questo punto il sistema fa previsioni su tutti gli utenti.

L'HR@n relativo ad un utente e' la percentuale dei primi n oggetti raccomandati per quell'utente
che effettivamente l'utente gradisce.
'''
'''
# all_users = [i for i in range(number_of_users)]
all_users = [i for i in range(10)]
# all_users = [42]

items_to_compute = [2, 4, 6, 8, 10, 12, 14]

# rec. train [219, 3324, 10, 983, 442, 1631, 225, 3437, 775, 2894]
# rec. test  [219, 3324, 10, 983, 19,  1631, 225, 3437, 20,  21] ????
# [219, 19, 20, 21, 3324, 3437, 983, 1631, 10, 225]
# [219, 19, 20, 21, 3324, 3437, 983, 1631, 10, 225]

#
number_of_processed_users = len(all_users)
print('Compute recom from TRAIN')
# recommended_items = CMN_wrapper_train.recommend(all_users, items_to_compute=items_to_compute, cutoff=cutoff)
recommended_items = CMN_wrapper_train.recommend(all_users, cutoff=cutoff)

# print('Compute recom from TEST')
# recommended_items_B = CMN_wrapper_test.recommend(all_users, items_to_compute=items_to_compute, cutoff=cutoff)

number_of_correct_answers = 0

for user_id in range(number_of_processed_users):
    print('{}/{}'.format((user_id + 1), number_of_processed_users))

    relevant_items = URM_test.indices[URM_test.indptr[user_id]:URM_test.indptr[user_id + 1]]

    is_relevant = np.in1d(recommended_items[user_id], relevant_items, assume_unique=True)

    number_of_correct_answers += is_relevant.sum()
    print('recommended_items train:', np.sort(recommended_items[user_id]))
    # print('recommended_items test:', np.sort(recommended_items[user_id]))
    print('relevant_items:', relevant_items)
    print('is_relevant:', is_relevant)
    print('number_of_correct_answers:', number_of_correct_answers)

ht = number_of_correct_answers / (cutoff * number_of_processed_users)
print('HR@{} = {}'.format(cutoff, ht))
'''
