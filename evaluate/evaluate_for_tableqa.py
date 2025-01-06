import argparse
import json
from collections import defaultdict
from collections import Counter
import numpy as np
import re


def _get_predict_list(_ground_str: list, target_delimiter=', '):
    # _ground_str中每一个list元素转换成功str
    # print(_ground_str)
    _ground_str = [ sample if type(sample) != list else ','.join(map(lambda x: str(x), sample)) for sample in _ground_str]
    # print(_ground_str)

    result = []
    _ground_spans = [x.lower().strip().strip('.').strip("'").strip('"').strip() for x in _ground_str]
    # print(_ground_spans)
    for i in range(len(_ground_spans)):
        if _ground_spans[i].endswith('.0'):
            _ground_spans[i] = _ground_spans[i][:-2]

        if _ground_spans[i].replace(',', '').isnumeric():
            _ground_spans[i] = _ground_spans[i].replace(',', '')
        
        result.append(_ground_spans[i]) 
    
    return result
    
    

def evaluate_example(_predict_str: str, _ground_str: list, target_delimiter=','):
    _predict_spans = _predict_str.split(target_delimiter)
    _predict_spans = [x.lower().strip().strip('.').strip("'").strip('"').strip() for x in _predict_spans]
    
    for i in range(len(_predict_spans)):
        if _predict_spans[i].endswith('.0'):
            _predict_spans[i] = _predict_spans[i][:-2]

        if _predict_spans[i].replace(',', '').isnumeric():
            _predict_spans[i] = _predict_spans[i].replace(',', '')

        _predict_spans[i] = _predict_spans[i].replace(' ', '')
    
    _ground_spans = [x.lower().strip().strip('.').strip("'").strip('"').strip() for x in _ground_str]
    
    for i in range(len(_ground_spans)):
        if _ground_spans[i].endswith('.0'):
            _ground_spans[i] = _ground_spans[i][:-2]

        if _ground_spans[i].replace(',', '').isnumeric():
            _ground_spans[i] = _ground_spans[i].replace(',', '')
        
        _ground_spans[i] = _ground_spans[i].replace(' ', '')

    _predict_values = defaultdict(lambda: 0)
    _ground_values = defaultdict(lambda: 0)
    for span in _predict_spans:
        try:
            _predict_values[float(span)] += 1
        except ValueError:
            _predict_values[span.strip()] += 1
    for span in _ground_spans:
        try:
            _ground_values[float(span)] += 1
        except ValueError:
            _ground_values[span.strip()] += 1
    _is_correct = _predict_values == _ground_values
    return _is_correct

def get_selfconsistency_denotation(prediction: list):
    # 得到prediction中出现次数最多的元素，如果有多个，返回第一个
    prediction = [['0'] if item == [] or item == 'None' or item == ['None'] or item == set() or item == [set()] or item == None or item == [None] or item == 'error' or item == ['error'] else item for item in prediction]
    prediction = [ str(item) if type(item) == int or type(item) == float else item for item in prediction]
    prediction = [ [item] if type(item) == str else item for item in prediction]
    prediction = _get_predict_list(prediction)
    # print(prediction)
    most_common_element = max(prediction, key=lambda x: prediction.count(x))
    if most_common_element == '0' and len(Counter(prediction)) > 1:
        element_count = Counter(prediction)
        return element_count.most_common(2)[-1][0], prediction.index(element_count.most_common(2)[-1][0])
    return most_common_element, prediction.index(most_common_element)

def evaluate_tableqa(prediction: str, label: list):
    # 预处理
    prediction = prediction.replace('\xa0', ' ')
    label = [str(item) if type(item)!=str else item for item in label]
    if label == ['0'] or label == ['None'] or label == ['error'] or label == ['null']: label = ['0']
    new_list = []

    if len(label)==1 and bool(re.fullmatch(r'[0-9,]*', label[0])):
        label[0] = label[0].replace(',','')

    for item in label:
        if "," in item:
            split_items = item.split(",")
            new_list.extend(split_items)
        else:
            new_list.append(item)
    
    return evaluate_example(prediction, new_list)

def evaluate(args):
    avg_deno_acc = []
    
    with open(args.ori_path, 'r') as f:
        for line in f:
            line = json.loads(line.strip())

            question = line[list(line.keys())[0]]['question']
            label = line[list(line.keys())[0]]['label']
            predictions = line[list(line.keys())[0]]['prediction']
            label = list(set(label))


            prediction, idx = get_selfconsistency_denotation(predictions)

            if evaluate_tableqa(prediction, label): #str list
                avg_deno_acc.append(1)
            else:
                if args.write_flag:
                    with open(args.error_cases_output, 'a') as f_error_cases:
                        f_error_cases.write(json.dumps({'question': question, 'label': label, 'prediction': prediction}) + '\n')
                avg_deno_acc.append(0)

    acc = np.mean(avg_deno_acc)
    print(len(avg_deno_acc))
    print("Denotation Acc: %.4f" % (acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_path', type=str, default="./output/wikisql/V3/final_output.txt")
    parser.add_argument('--error_cases_output', type=str,
                        default='./output/wikisql/bad_cases.txt')
    parser.add_argument('--write_flag', type=bool, default=False)
    args = parser.parse_args()

    evaluate(args)