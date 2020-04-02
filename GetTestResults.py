import pickle

import os

# root = "result_experiments/KDD/MCRec_movielens100k"
# cutoff = 10
# root = "result_experiments/KDD/CollaborativeVAE_citeulike_a_1"
root = "result_experiments/SIGIR/CMN_pinterest"
cutoff = 5

print_metrics_names = True

table = {}
table[0, 0] = 'Metrics@{}'.format(cutoff)
column = 1
selected_metrics = {'CUSTOM_HIT_RATE',
                    'HIT_RATE', 'NDCG',
                    'WEIGHTED_HIT_RATE',
                    'POS_WEIGHTED_HIT_RATE',
                    'LOG_WEIGHTED_HIT_RATE',
                    'LOG_POS_WEIGHTED_HIT_RATE'
                    }

for file in os.listdir(root):
    if file.endswith("_metadata"):
        f = os.path.join(root, file)

        result_dict = pickle.load(open(f, "rb"))
        # print(result_dict)
        # print(result_dict['best_result_test'])
        # print(result_dict['best_result_validation'])
        # best_result = result_dict['best_result_validation']
        best_result = result_dict['best_result_test'][cutoff]

        if file[:-9] == 'CMN_RecommenderWrapper':
            print(best_result)

        if print_metrics_names:
            row = 1
            for item in best_result:
                # print(item)
                if item in selected_metrics:
                    table[row, 0] = str(item)
                    row += 1

            print_metrics_names = False
            # print('\n')

        # print('\n{}'.format(file[:-9]))
        table[0, column] = file[:-9]

        row = 1

        for item in best_result:
            # print(round(best_result[item], 4))
            if item in selected_metrics:
                print(file[:-9], item, table[row, 0])
                assert item == table[row, 0]
                table[row, column] = str(round(best_result[item], 4))
                row += 1

        column += 1

print('rows:', row)
print('columns:', column)
# print(table)
for r in range(row):
    for c in range(column):
        print(table[r, c], end='\t')
        # pass
    print('')
