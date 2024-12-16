import json 
from tqdm import tqdm
import argparse
import os
import structllm as sllm
import structllm.translate2CGdata as trans2CGdata
import random
import multiprocessing as mp
import sys
from collections import defaultdict
import CGdata_for_tableqa

def TEMP_Question_Answer(args, all_data, idx, api_key, table_data):
    os.environ["OPENAI_BASE_URL"] = args.openai_url
    os.environ["OPENAI_API_KEY"] = args.key
    
    collection = None
    if args.retriever is not None:
        collection = sllm.retrieve.get_embedding(args.data_path, args.retriever, args.chroma_dir)
    
    # openai.api_key = api_key
    args.key = api_key
    # print(f"args.key:{args.key}, idx:{idx}")
    # import pdb; pdb.set_trace()

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
            for item in tqdm(all_data, total=len(all_data), desc="PID: %s" % idx):
                question = item.question
                answer = item.answer

                fdetail.write(f"********* Table_temp *********\n")
                fdetail.write(f"=== Question:{question}\n")
                fdetail.write(f"=== Answer:{answer}\n")
                
                if not args.debug:
                    try:
                        sys.stdout = fdetail
                        result, query_list, prompt_list = sllm.tempqa.tempqa(args, item, table_data, collection)
                        sys.stdout = sys.__stdout__
                        fdetail.write(f"=== Answer:{answer}\n")
                        fdetail.write(f"=== Result:{result}\n")
                        # result = [ list(sample) if type(sample)==set else sample for sample in tmp_result ]
                        print(f"label:{answer}, result:{result}, output_result_path:{output_result_path}, answer_type:{item.answer_type}, type:{item._type}")
                        result_dict = dict()
                        tmp_dict = {"question":question,"label":answer,"prediction":result,"answer_type":item.answer_type,"type":item._type}
                        result_dict = tmp_dict
                        fresult.write(json.dumps(result_dict) + "\n")
                        fdetail.write(json.dumps(result_dict) + "\n")
                        fdetail.flush()
                        fresult.flush()

                    except Exception as e:    
                        tmp_dict = {"question":question, "answer": answer, "error": str(e)}
                        if args.store_error:
                            error_path = os.path.join(output_detail_path[:output_detail_path.rfind("/")], args.error_file_path)
                            with open(error_path, "a") as f:
                                f.write(json.dumps(tmp_dict) + "\n")

                else:
                    # sys.stdout = fdetail
                    result, query_list, prompt_list = sllm.tempqa.tempqa(args, item, table_data, collection)
                    # sys.stdout = sys.__stdout__
                    fdetail.write(f"=== Answer:{answer}\n")
                    fdetail.write(f"=== Result:{result}\n")
                    # import pdb; pdb.set_trace()
                    # result = [ list(sample) if type(sample)==set else sample for sample in tmp_result ]
                    print(f"label:{answer}, result:{result}, output_result_path:{output_result_path}, answer_type:{item.answer_type}, type:{item._type}")
                    result_dict = dict()
                    tmp_dict = {"question":question,"label":answer,"prediction":result,"answer_type":item.answer_type,"type":item._type}
                    result_dict = tmp_dict
                    fresult.write(json.dumps(result_dict) + "\n")
                    fdetail.write(json.dumps(result_dict) + "\n")
                    fdetail.flush()
                    fresult.flush()


if __name__=="__main__":
    args = CGdata_for_tableqa.parse_args()
    print(f"SC_Num:{args.SC_Num}\n")
    # get API key
    if not args.key.startswith("sk-"):
        with open(args.key, "r") as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
            assert len(all_keys) >= args.num_process, (len(all_keys), args.num_process)
   
    # get data
    KG_data, KGQA_data = trans2CGdata.read_tempKG(args) # get CGdata and QAdata

    if args.num_process == 1:
        TEMP_Question_Answer(args, KGQA_data, -1, args.key, KG_data)
    else:
        num_each_split = int(len(KGQA_data) / args.num_process)
        p = mp.Pool(args.num_process)
        for idx in range(args.num_process):
            start = idx * num_each_split
            if idx == args.num_process - 1:
                end = max((idx + 1) * num_each_split, len(KGQA_data))
            else:
                end = (idx + 1) * num_each_split
            split_data = KGQA_data[start:end]
            p.apply_async(TEMP_Question_Answer, args=(args, split_data, idx, all_keys[idx], KG_data))
        p.close()
        p.join()
        print("All of the child processes over!")