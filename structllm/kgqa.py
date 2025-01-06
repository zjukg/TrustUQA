import re
import structllm as sllm
import openai
from openai import OpenAI

def kgqa(args, question, table_data, relations, collection=None): 

    llm = sllm.llm.gpt(args)

    query_prompt = sllm.prompt.kgqa_query_prompt(args, question, table_data, relations, collection)
    print(query_prompt.naive_prompt[-1])

    result_list = []        # result list
    query_list  = []        # query  list
    prompt_list = []        # prompt list

    max_retries = 3         
    retry_count = 0         
    SC_Num = args.SC_Num    
    total_num = 0           
        
    while retry_count < max_retries and SC_Num >0 and total_num < max_retries*SC_Num:
        retry_count += 1

        print(f"Retry_count:{retry_count}, Num of SC:{SC_Num}, Total_numL:{total_num}")
        
        '''question -> query'''
        responses = llm.get_response(query_prompt.naive_prompt, flag = 1, num = SC_Num)        
            
        for response in responses:
            try:
                response = response.message.content
                print("      ###1.generated query_text:", response)
                # Step2: Get target_type from response
                type2id, target_type = sllm.align.get_target_type(args, response, table_data)
                print("      ###2.generated target_type:", target_type)

                # Step3: parameter retrieval and replacement
                text_query, id_query, step_query = sllm.align.MetaQA_text2query(args, response, question, table_data, relations)
                print("      ###3.retrieved parameters(node):", text_query)
                print("      ###3.retrieved parameters(id):", id_query)

                '''query -> result'''
                # execute query
                print("      ###4.excute process:")
                if target_type == None: res, mid_output = table_data.excute_query(args, id_query, target_type=None, node_query=text_query, task=step_query, question=question)
                else: res, mid_output = table_data.excute_query(args, id_query, target_type=target_type[0], node_query=text_query, task=step_query, question=question)
                
            except openai.BadRequestError as e: # 非法输入 '$.input' is invalid. query返回结果为：请输入详细信息等 
                print(e)
                total_num += 1
                continue

            except IndexError as e: # 得不到正确格式的query: set1=(fastest car)
                print(e)
                total_num += 1 # 防止卡死
                continue

            except openai.APITimeoutError as e: # 超时
                print(e)
                total_num += 1 # 防止卡死
                continue

            except ValueError as e: # maximum context length
                print(e)
                continue

            except Exception as e: # 其他错误
                print(e)
                continue
            
            total_num += 1

            if res == None or res == [] or res == set() or res == [set()] or res == dict() or res == [set([])] or res == ['None'] or res == ['none'] or (type(res)==str and "[line_" in res ):
                print("      ###5.excute result is None, retry...")
                if retry_count >= max_retries: result_list.append(res)
                continue
            else:
                print("      ###5.excute result:",res)
                SC_Num -= 1
                result_list.append(res)
                query_list.append(text_query)
                prompt_list.append(response)

    while len(result_list) < args.SC_Num: result_list.append("0")
    print("###6.final result", result_list)

    result = [ list(sample) if type(sample)==set else sample for sample in result_list ]
    return result, query_list, prompt_list