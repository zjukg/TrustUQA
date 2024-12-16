import argparse
import json
from collections import defaultdict
import numpy as np
from collections import Counter

def get_selfconsistency_res(prediction: list):
    def find_most_common_except_LineAndZero(prediction):
        # 使用 Counter 统计元素出现次数
        counter = Counter(tuple(sublist) for sublist in prediction)
        most_common_elements = counter.most_common()
        # 检查列表长度和第一个元素是否是 '0'
        if len(most_common_elements) == 1 and most_common_elements[0][0] == ('0',):
            return prediction[0], 0
        else:
            if(most_common_elements[0][0] == ('0',)): 
                res = list(most_common_elements[1][0]) #如果最大是0返回次大
            else: 
                res = list(most_common_elements[0][0]) #返回最大值
        return res, prediction.index(res)

    # 得到prediction中出现次数最多的元素，如果有多个，返回第一个
    prediction = [['0'] if item == [] or item == 'None' or item == ['None'] or item == set() or item == [set()] or item == None or item == [None] or item == 'error' or item == ['error'] else item for item in prediction]
    prediction = [ str(item) if type(item) == int or type(item) == float else item for item in prediction]
    prediction = [ [item] if type(item) == str else item for item in prediction]
    
    return find_most_common_except_LineAndZero(prediction)

def evaluate_metaqa(prediction, label):
    # if set(label).issubset(prediction):
    if prediction[0] in label:
        return 1
    else:
        return 0

def evaluate(args):
    avg_deno_acc = []

    # question_list = []
    # with open("dynamic_final/metaqa/1hop_template_50/all_result.txt", "r") as f:
    #     for line in f:
    #         line = json.loads(line.strip())
    #         question = line[list(line.keys())[0]]['question']
    #         question_list.append(question)
            

    with open(args.ori_path, 'r') as f:
        for line in f:
            line = json.loads(line.strip())

            question = line[list(line.keys())[0]]['question']
            label = line[list(line.keys())[0]]['label']
            predictions = line[list(line.keys())[0]]['prediction']

            # if(question not in question_list): continue

            prediction, idx = get_selfconsistency_res(predictions)

            if evaluate_metaqa(prediction, label):
                avg_deno_acc.append(1)
            else:
                if args.write_flag:
                    with open(args.error_cases_output, 'a') as f_error_cases:
                        f_error_cases.write(json.dumps({'question': question, 'label': label, 'prediction': prediction}) + '\n')
                avg_deno_acc.append(0)

    print(len(avg_deno_acc))
    acc = np.mean(avg_deno_acc)
    print("Acc: %.4f" % (acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_path', type=str, default="./output/metaqa/all_result.txt")
    parser.add_argument('--error_cases_output', type=str,
                        default='./output/metaqa/bad_cases_tmp.txt')
    parser.add_argument('--write_flag', type=bool, default=True)
    args = parser.parse_args()

    evaluate(args)