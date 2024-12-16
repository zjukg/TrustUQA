import argparse
import json
from collections import defaultdict
import numpy as np

def get_selfconsistency_res(prediction: list):
    def find_most_common_except_LineAndZero(prediction):
        if not prediction:
            return '0'
        most_common_element = max(prediction, key=lambda x: prediction.count(x))
        # print(f"most_common_element:{most_common_element}")
        if most_common_element == ['0']:
            prediction = [x for x in prediction if x != most_common_element]
            return find_most_common_except_LineAndZero(prediction)
        else:
            return most_common_element

    # 得到prediction中出现次数最多的元素，如果有多个，返回第一个
    prediction = [['0'] if item == [] or item == 'None' or item == ['None'] or item == set() or item == [set()] or item == None or item == [None] or item == 'error' or item == ['error'] else item for item in prediction]
    prediction = [ str(item) if type(item) == int or type(item) == float else item for item in prediction]
    prediction = [ [item] if type(item) == str else item for item in prediction]
    
    return find_most_common_except_LineAndZero(prediction)

def evaluate_cron(prediction, label):
    label = [item if type(item) == str else str(item) for item in label]

    prediction = [item if type(item) == str else str(item) for item in prediction]

    for i in range(len(prediction)):
        if prediction[i].endswith('.0'):
            prediction[i] = prediction[i][:-2]

    def is_all_digits_or_dash(s):
        return all(c.isdigit() or c == '-' or c == ',' or c == ' ' for c in s)

    final_prediction = prediction[0]
    if type(final_prediction) == list:
        final_prediction = final_prediction[0]

    if is_all_digits_or_dash(final_prediction):
        final_prediction = final_prediction.split(',')[0]

    
    return final_prediction in set(label)

def evaluate(args):
    avg_deno_acc = []

    entity_acc = []
    time_acc = []

    simple_acc = []
    complex_acc = []

    # question_list = []
    # with open("dynamic_output/cron/test100_TopK_DPR/all_result.txt", 'r') as f:
    #     for line in f:
    #         line = json.loads(line.strip())
    #         # import pdb; pdb.set_trace()
    #         question = line['question']
    #         question_list.append(question)

    with open(args.ori_path, 'r') as f:
        for line in f:
            line = json.loads(line.strip())
            # print(line.keys())
            question = line['question']
            label = line['label']
            prediction = line['prediction']
            _type = line['type']
            answer_type = line['answer_type']

            prediction = get_selfconsistency_res(prediction)
            
            # if question not in question_list: continue

            flag = 0
            if evaluate_cron(prediction, label): # final_prediction in set(label):
                flag = 1
            else:
                if args.write_flag:
                    with open(args.error_cases_output, 'a') as f_error_cases:
                        f_error_cases.write(json.dumps({'question': question, 'label': label, 'prediction': prediction}) + '\n')

            avg_deno_acc.append(flag)

            if _type == 'simple_time' or _type == 'simple_entity':
                simple_acc.append(flag)
            else:
                complex_acc.append(flag)

            if answer_type == 'time':
                time_acc.append(flag)
            else:
                entity_acc.append(flag)

    print(len(avg_deno_acc))
    acc = np.mean(avg_deno_acc)
    print("overall_acc: %.4f" % (acc))

    simple_acc = np.mean(simple_acc)
    print("simple_acc: %.4f" % (simple_acc))

    complex_acc = np.mean(complex_acc)
    print("complex_acc: %.4f" % (complex_acc))

    time_acc = np.mean(time_acc)
    print("time_acc: %.4f" % (time_acc))

    entity_acc = np.mean(entity_acc)
    print("entity_acc: %.4f" % (entity_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_path', type=str, default="./output/metaqa/all_result.txt")
    parser.add_argument('--error_cases_output', type=str,
                        default='./output/metaqa/bad_cases.txt')
    parser.add_argument('--write_flag', type=bool, default=False)
    args = parser.parse_args()

    evaluate(args)