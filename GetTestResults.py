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

for file in os.listdir(root):
    if file.endswith("_metadata"):
        f = os.path.join(root, file)

        result_dict = pickle.load(open(f, "rb"))
        #print(result_dict)
        #print(result_dict['best_result_test'])
        #print(result_dict['best_result_validation'])
        # best_result = result_dict['best_result_validation']
        best_result = result_dict['best_result_test'][cutoff]

        if print_metrics_names:
            # print("Metrics")
            row = 1
            for item in best_result:
                # print(item)
                table[row, 0] = str(item)
                row += 1

            print_metrics_names = False
            # print('\n')

        # print('\n{}'.format(file[:-9]))
        table[0, column] = file[:-9]

        row = 1
        for item in best_result:
            # print(round(best_result[item], 4))
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
