import json 
from tqdm import tqdm
import argparse
import os
import structllm as sllm
import random
import multiprocessing as mp
import sys
# import openai

def TableID_Question_Answer(args, all_data, idx, api_key, table_data, collection = None):
    os.environ["OPENAI_BASE_URL"] = args.openai_url
    os.environ["OPENAI_API_KEY"] = args.key
    collection = None
    if args.retriever is not None:
        collection = sllm.retrieve.get_embedding(args.data_path, args.retriever, args.chroma_dir)
    
    # openai.api_key = api_key
    args.key = api_key
    # print(f"args.key:{args.key}, idx:{idx}")

    if idx == -1:
        output_detail_path = args.output_detail_path
        output_result_path = args.output_result_path
    else:
        idx = "0" + str(idx) if idx < 10 else str(idx)  # 00 01 02 ... 29
        output_detail_path = args.output_detail_path + "_" + idx
        output_result_path = args.output_result_path + "_" + idx

    print("Start PID %d and save to %s" % (os.getpid(), output_result_path))

    with open(output_result_path+".txt", "w") as fresult:
        with open(output_detail_path+".txt", "w") as fdetail:
            # for (table_id, question, answer) in tqdm(all_data, total=len(all_data), desc="PID: %d" % os.getpid()):
            for (table_id, question, answer) in tqdm(all_data, total=len(all_data), desc="PID: %s" % idx):
                fdetail.write(f"********* Table{table_id} *********\n")
                fdetail.write(f"=== Question:{question}\n")
                fdetail.write(f"=== Answer:{answer}\n")
                
                if not args.debug:
                    try:
                        sys.stdout = fdetail
                        result, query_list, prompt_list = sllm.tableqa.tableqa(args, question, table_data[table_id], collection=collection)
                        sys.stdout = sys.__stdout__
                        fdetail.write(f"=== Answer:{answer}\n")
                        fdetail.write(f"=== Result:{result}\n")
                        print(f"label:{answer}, result:{result}, output_result_path:{output_result_path}")
                        result_dict = dict()
                        tmp_dict = {"question":question,"label":answer,"prediction":result}
                        result_dict[table_id] = tmp_dict
                        fresult.write(json.dumps(result_dict) + "\n")
                        fdetail.write(json.dumps(result_dict) + "\n")
                        fdetail.flush()
                        fresult.flush()

                    except Exception as e:    
                        tmp_dict = {"tableid":table_id, "question":question, "answer": answer, "error": str(e)}
                        if args.store_error:
                            error_path = os.path.join(output_detail_path[:output_detail_path.rfind("/")], args.error_file_path)
                            with open(error_path, "a") as f:
                                f.write(json.dumps(tmp_dict) + "\n")

                else:
                    # sys.stdout = fdetail
                    result, query_list, prompt_list = sllm.tableqa.tableqa(args, question, table_data[table_id], collection=collection)
                    # sys.stdout = sys.__stdout__
                    fdetail.write(f"=== Answer:{answer}\n")
                    fdetail.write(f"=== Result:{result}\n")
                    # result = [ list(sample) if type(sample)==set else sample for sample in result ]
                    print(f"label:{answer}, result:{result}, output_result_path:{output_result_path}")
                    result_dict = dict()
                    tmp_dict = {"question":question,"label":answer,"prediction":result}
                    result_dict[table_id] = tmp_dict
                    fresult.write(json.dumps(result_dict) + "\n")
                    fdetail.write(json.dumps(result_dict) + "\n")
                    fdetail.flush()
                    fresult.flush()

def get_cgdata(folder_path, error_file_list):
    table_data = dict()
    file_names = os.listdir(folder_path) # os.listdir()

    paths_of_files = [] # all csv path
    for file_name in file_names:
        path_of_file = os.path.join(folder_path, file_name)
        if not os.path.isdir(path_of_file):
            paths_of_files.append(path_of_file)
        else:
            for child_file in os.listdir(path_of_file):
                paths_of_files.append(os.path.join(path_of_file, child_file))
        
    # error_file_list = []
    with open("error_file.txt", "w") as f:
        for path_of_file in tqdm(paths_of_files):
            if path_of_file.count("/") != 4: 
                table_name = path_of_file.split('/')[-1].split('.')[0]
            else: 
                table_name = path_of_file[path_of_file.find('/', path_of_file.find('/') + 1) + 1:].split('.')[0]
            
            try:
                test_table_ = sllm.translate2CGdata.csv2CG(path_of_file)
                table_data[table_name] = sllm.cg.data(test_table_['triples'], test_table_['entities_2_line'], test_table_['all_lines_id'])
            except:
                f.write(table_name)
                error_file_list.append(table_name+'.csv')
                f.write("\n")
    
    return table_data


def read_table_data(args):
    print('read table data...')

    error_file_list = []
    table_data = get_cgdata(args.folder_path, error_file_list)
    
    if "wikisql" in args.folder_path.lower() and args.retriever is not None:
        # assert args.train_folder_path != None, "train_folder_path is None"
        train_table_data = get_cgdata(args.folder_path.replace("test", "train"), error_file_list)
        table_data.update(train_table_data)
        dev_table_data = get_cgdata(args.folder_path.replace("test", "dev"), error_file_list)
        table_data.update(dev_table_data)
    
    with open(args.data_path, 'r') as fp:
        tb_question = json.loads(fp.read())

    with open("error_file.txt", "w") as f:
        for k, v in tb_question.items():
            if k in error_file_list:
                f.write(k)
                f.write("\n")

    tb_question = {k.split('.')[0]: v for k, v in tb_question.items() if k not in error_file_list}

    TableQA_data = []
    for table_id in tb_question.keys():
        for qa in tb_question[table_id]:
            question = qa[0]
            answer = qa[1]
            TableQA_data.append((table_id, question, answer))

    return table_data, TableQA_data

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    
    # setting for openai
    parser.add_argument('--openai_url', default="", type=str, help='The url of openai')
    parser.add_argument('--key', default="", type=str, help='The key of openai or path of keys')

    # setting for alignment retriever
    parser.add_argument('--retriever_align', default="OpenAI", type=str, help='The retriever used for alignment')
    
    # input data path
    parser.add_argument('--folder_path', default="dataset/WikiSQL_TB_csv/test", type=str, help='The CSV data pth.')
    parser.add_argument('--data_path', default="dataset/WikiSQL_CG", type=str, help='The CG data pth.')
    parser.add_argument('--prompt_path', default="structllm/prompt_/wikisql.json", type=str, help='The prompt pth.')
    
    # setting for model and sc
    parser.add_argument('--SC_Num', default=5, type=int)
    parser.add_argument('--model', default="gpt-3.5-turbo-0613", type=str, help='The openai model. "gpt-3.5-turbo-0613" and "gpt-4-1106-preview" are supported')
        
    # output
    parser.add_argument('--store_error', action="store_true", default=True)
    parser.add_argument('--error_file_path', default="timeout_file.txt", type=str)
    parser.add_argument('--output_detail_path', default="output/V3/output_detail", type=str)
    parser.add_argument('--output_result_path', default="output/V3/output_result", type=str)
    
    # setting for dynamic prompt
    parser.add_argument('--retriever', default=None, type=str, help='The retriever used for few-shot retrieval')
    # parser.add_argument('--train_folder_path', default=None, type=str, help='The train folder path for few-shot retrieval')
    parser.add_argument('--chroma_dir', default="chroma", type=str, help='The chroma dir.')
    parser.add_argument('--retrieved_history', default="structllm/prompt_/train_candidate_demo_wtq.json", type=str, help='The path of candidate demo in train or dev')
    parser.add_argument('--dynamically_prompt_num', default=8, type=int, help='The number of dynamical prompts for each question')
    parser.add_argument('--sampling', default="TopK", type=str, help='sampling method for dynamical prompt, "TopK", "Random", "Beta" or "Exponential"')
    parser.add_argument('--sampling_num', default=15, type=int, help='The number of sampling for each question')
    

    #others
    parser.add_argument('--num_process', default=1, type=int, help='the number of multi-process')
    parser.add_argument('--debug', default=0, type=int)
    
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    # os.environ["OPENAI_BASE_URL"] = args.openai_url
    
    print(f"SC_Num:{args.SC_Num}\n")
    # get API key
    if not args.key.startswith("sk-"):
        with open(args.key, "r") as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
            assert len(all_keys) >= args.num_process, (len(all_keys), args.num_process)
   
    # get data
    table_data, TableQA_data = read_table_data(args) # get CGdata and QAdata
    args.table_data = table_data
    
    if args.num_process == 1:
        TableID_Question_Answer(args, TableQA_data, -1, args.key, table_data)
    else:
        num_each_split = int(len(TableQA_data) / args.num_process)
        p = mp.Pool(args.num_process)
        for idx in range(args.num_process):
            start = idx * num_each_split
            if idx == args.num_process - 1:
                end = max((idx + 1) * num_each_split, len(TableQA_data))
            else:
                end = (idx + 1) * num_each_split
            split_data = TableQA_data[start:end]
            p.apply_async(TableID_Question_Answer, args=(args, split_data, idx, all_keys[idx], table_data))
        p.close()
        p.join()
        print("All of the child processes over!")

        
