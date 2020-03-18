import pickle

import os

root = "result_experiments/KDD/MCRec_movielens100k"
print_metrics_names = True

for file in os.listdir(root):
    if file.endswith("_metadata"):
        f = os.path.join(root, file)
        if file=='TopPopRecommender_metadata':

            result_dict = pickle.load(open(f, "rb"))
            print(result_dict)
            # print(result_dict['best_result_validation'])
