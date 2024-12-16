import argparse
import json
from collections import defaultdict
import numpy as np    
import evaluate_for_metaqa as metaqa

# def get_selfconsistency_res(prediction: list):
#     def find_most_common_except_LineAndZero(prediction):
#         if not prediction:
#             return '0'
#         most_common_element = max(prediction, key=lambda x: prediction.count(x))
#         # print(f"most_common_element:{most_common_element}")
#         if most_common_element == ['0']:
#             prediction = [x for x in prediction if x != most_common_element]
#             return find_most_common_except_LineAndZero(prediction)
#         else:
#             return most_common_element

#     prediction = [['0'] if item == [] or item == 'None' or item == ['None'] or item == set() or item == [set()] or item == None or item == [None] or item == 'error' or item == ['error'] else item for item in prediction]
#     prediction = [ str(item) if type(item) == int or type(item) == float else item for item in prediction]
#     prediction = [ [item] if type(item) == str else item for item in prediction]
    
#     return find_most_common_except_LineAndZero(prediction)

def evaluate_wqsp(prediction, label):
    # if set([prediction[0]]).issubset(label):
    if set(label).issubset(prediction):
        return 1
    else:
        return 0


def evaluate(args):
    avg_deno_acc = []
    with open(args.ori_path, 'r') as f:
        for line in f:
            line = json.loads(line.strip())

            question = line[list(line.keys())[0]]['question']
            question = question.split('\n[')[0]
            label = line[list(line.keys())[0]]['label']
            prediction = line[list(line.keys())[0]]['prediction']

            prediction, idx = metaqa.get_selfconsistency_res(prediction)

            if evaluate_wqsp(prediction, label):
                avg_deno_acc.append(1)
            else:
                if args.write_flag:
                    with open(args.error_cases_output, 'a') as f_error_cases:
                        f_error_cases.write(json.dumps({'question': question, 'label': label, 'prediction': prediction}) + '\n')
                avg_deno_acc.append(0)

    acc = np.mean(avg_deno_acc)
    print("Acc: %.4f" % (acc))
    print(len(avg_deno_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_path', type=str, default="./output/metaqa/all_result.txt")
    parser.add_argument('--error_cases_output', type=str,
                        default='./output/metaqa/bad_cases.txt')
    parser.add_argument('--write_flag', type=bool, default=False)
    args = parser.parse_args()

    evaluate(args)