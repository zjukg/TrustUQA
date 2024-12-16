import os
import re
import json
import fcntl
import random
import numpy as np
import structllm as sllm
import structllm.translate2CGdata as trans2CGdata
import evaluate.evaluate_for_CronQuestion as Cron
import evaluate.evaluate_for_wqsp as Wqsp
from evaluate.evaluate_for_metaqa import get_selfconsistency_res, evaluate_metaqa
from evaluate.evaluate_for_tableqa import get_selfconsistency_denotation, evaluate_tableqa

def get_beta_sample(count, n):
    result = []
    while len(result) < n:
        alpha=0.25
        beta=0.25
        sampled_values = np.random.beta(alpha, beta, size=n)
        sampled = [int(item*(count-1))for item in sampled_values] # 映射
        for item in sampled:
            if item not in result:
                result.append(item)

    return result[:n]

def get_exponential_sample(count, n):
    result = []
    while len(result) < n:
        lambda_param = 0.1
        sampled_values = np.random.exponential(scale=1/lambda_param, size=25)
        sampled = [int(item) for item in sampled_values]
        for item in sampled:
            if item not in result:
                result.append(item)

    print(result[:n])
    return result[:n]

def Get_Sampling_Results(args, question, collection):
    if args.sampling == "TopK":
        return collection.query(query_texts=[question], n_results=args.sampling_num)
    else:
        total_num = collection.count()
        if args.retriever == 'ANCE' or args.retriever == 'SentenceBERT' or args.retriever == 'text-embedding-ada-002':
            total_num -= 100
        tmp_result = collection.query(query_texts=[question], n_results=total_num)
        
        if args.sampling == "Random":
            sample_list = random.sample(range(total_num), args.sampling_num)
        elif args.sampling == "Beta": 
            sample_list = get_beta_sample(total_num, args.sampling_num)
        elif args.sampling == "Exponential":
            sample_list = get_exponential_sample(total_num, args.sampling_num)
        else:
            raise ValueError("Sampling method not supported")
        
        for keys in tmp_result.keys():
            if keys == 'embeddings' or keys == 'uris' or keys == 'data': continue
            tmp_result[keys][0] = [tmp_result[keys][0][i] for i in sample_list]

        return tmp_result

    

class query_prompt():
    def __init__(self, args, question, table_data, collection = None):
        self.question = question
        self.model = args.model
        if collection is not None:
            self.retrieve_dynamic_prompt(args, question, table_data, collection)                    
        else:
            with open(args.prompt_path, 'r') as json_file:
                self.naive_prompt = json.load(json_file)
        self.naive_prompt.append(
            {
                "role": "user",
                "content": self.schema_Prompt(table_data, question)
            }
        )

    def retrieve_dynamic_prompt(self, args, question, table_data, collection):
        dynamic_prompt = []
        success_num = 0
        
        history_train = {}
        if os.path.exists(args.retrieved_history) and collection is not None:
            with open(args.retrieved_history, 'r') as json_file:
                history_train = json.load(json_file)

        # results = collection.query(query_texts=[question], n_results=15)
        results = Get_Sampling_Results(args, question, collection)
        # results = collection.query(query_texts=[question], n_results=collection.count())
        assert results['metadatas'][0][0]['source'] != 'test', "Dynamic prompt cannot be generated using the test set"

        for idx in range(len(results['documents'][0])):
            print(f"Question: {question}.\nRetrieve for {idx}")
            if(success_num == args.dynamically_prompt_num): break

            temp_dynamic_prompt = None
            retrieved_question = results['documents'][0][idx]
            retrieved_tabledata = args.table_data[results['metadatas'][0][idx]['table']]

            if retrieved_question in history_train.keys():
                if(history_train[retrieved_question]=="None"): 
                    continue

                print(f"[{retrieved_question}] Retrieve from history successfully")
                temp_dynamic_prompt = {
                    "question": retrieved_question,
                    "prompt": history_train[retrieved_question],
                    "table_data": retrieved_tabledata
                }
            else:
                tmp_answer = results['metadatas'][0][idx]['answer']
                answer = tmp_answer.split("|")
                # try:
                result_list, _, prompt_list = sllm.tableqa.tableqa(args, retrieved_question, retrieved_tabledata)
                for _idx, prompt in enumerate(prompt_list):
                    prediction, _ = get_selfconsistency_denotation([result_list[_idx]])
                    if(evaluate_tableqa(prediction, answer)):
                        print(f"Retrieve for {retrieved_question} successfully")
                        temp_dynamic_prompt = {
                            "question":retrieved_question,
                            "prompt": prompt,
                            "table_data": retrieved_tabledata
                        }
                        history_train[retrieved_question] = prompt_list[_idx]
                        # with open(args.retrieved_history, 'w') as json_file:
                        #     json.dump(history_train, json_file, indent=4)
                        print(f"{retrieved_question}: {prompt_list[_idx]}")
                        break

                # except Exception as e:
                #     print(f"Error in {retrieved_question}: {e}")
                #     continue

            if temp_dynamic_prompt is not None:
                success_num += 1
                print(f"success_num: {success_num} retrieved_question:{retrieved_question}")
                dynamic_prompt.append(temp_dynamic_prompt)
                
            else:
                history_train[retrieved_question] = "None"
                # with open(args.retrieved_history, 'w') as json_file: # save the error retrieved_pair
                #     json.dump(history_train, json_file, indent=4)

        if success_num == 0:
            print(f"For \"{question}\", only {success_num} prompts are successfully retrieved, less than {args.dynamically_prompt_num} prompts are retrieved.\nUse the naive prompt instead.\n")
            with open(args.prompt_path, 'r') as json_file:
                self.naive_prompt = json.load(json_file)
        elif success_num == args.dynamically_prompt_num:
            print(f"@@@@@@@@\"{question}\", {success_num} prompts are successfully retrieved.\n")
            self.naive_prompt = self.build_dynamic_prompt(args, dynamic_prompt)
        
        else:
            self.naive_prompt = self.build_dynamic_prompt(args, dynamic_prompt)
            with open(args.prompt_path, 'r') as json_file:
                static_prompt = json.load(json_file)
            
            print(f"@@@@@@@@\"{question}\", only {success_num} prompts are successfully retrieved, less than {args.dynamically_prompt_num} prompts are retrieved.\n")
            for i in range(args.dynamically_prompt_num - success_num):
                self.naive_prompt.append(static_prompt[-(i*2+2)])
                self.naive_prompt.append(static_prompt[-(i*2+1)])

        with open(args.retrieved_history, 'r+') as json_file:
            fcntl.flock(json_file, fcntl.LOCK_EX)  # get exclusive lock

            existing_data = json.load(json_file)
            json_file.seek(0)  # move the file pointer to the beginning of the file

            for key in existing_data.keys():
                if key not in history_train.keys() or history_train[key] == "None":
                    history_train[key] = existing_data[key]

            json.dump(history_train, json_file, indent=4)  #  write the updated content
            json_file.truncate()  # truncate the extra content
            fcntl.flock(json_file, fcntl.LOCK_UN)  # release lock



    def build_dynamic_prompt(self, args, dynamic_prompt):
        prompt = []
        # get the instruction
        with open(args.prompt_path, 'r') as json_file:
            prompt.append(json.load(json_file)[0])
        
        for item in dynamic_prompt:
            prompt.append(
                {
                    "role": "user",
                    "content": self.schema_Prompt(item['table_data'], item['question'])
                }
            )
            prompt.append(
                {
                    "role": "assistant", 
                    "content": item['prompt']
                }
            )

        return prompt

    def schema_Prompt(self, table_data, question):
        relations, values = sllm.align.get_schema(table_data)
        
        tmp_prompt = str()
        for idx,item in enumerate(relations):
            if idx == 0:
                tmp_prompt += f"{item}:{values[idx]}"
            else:
                tmp_prompt += f"|{item}:{values[idx]}"

        prompt = f"Schema: {tmp_prompt}.\nQuestion: {question}"
        return prompt


class kgqa_query_prompt():
    def __init__(self, args, question, table_data, relations, collection = None):
        self.question = question
        self.model = args.model
        
        # retrieve demonstrations
        if collection is not None:
            self.retrieve_dynamic_prompt(args, question, table_data, relations, collection)                    
        else:
            with open(args.prompt_path, 'r') as json_file:
                self.naive_prompt = json.load(json_file)
        
        self.naive_prompt.append(
            {
                "role": "user",
                "content": self.kgqa_schema_Prompt()
            }
        )

    def kgqa_schema_Prompt(self):
        
        prompt = f"Question: {self.question}"
        return prompt
    
    def retrieve_dynamic_prompt(self, args, question, table_data, relations, collection):
        dynamic_prompt = []
        success_num = 0
        history_train = {}
        if os.path.exists(args.retrieved_history) and collection is not None:
            with open(args.retrieved_history, 'r') as json_file:
                history_train = json.load(json_file)

        template = re.sub(r'\[.*?\]', '[]', question)
        results = collection.query(query_texts=[question], n_results=15, where={"template":template})
        assert results['metadatas'][0][0]['source'] != 'test', "Dynamic prompt cannot be generated using the test set"

        for idx in range(len(results['documents'][0])):
            print(f"Question: {question}.\nRetrieve for {idx}")
            if(success_num == args.dynamically_prompt_num): break

            temp_dynamic_prompt = None
            retrieved_question = results['documents'][0][idx]

            # topic = results['metadatas'][0][idx]['topic']
            # retrieved_question = processed_question.replace('[]', topic)

            if retrieved_question in history_train.keys():
                if(history_train[retrieved_question]=="None"): 
                    continue

                print(f"[{retrieved_question}] Retrieve from history successfully")
                temp_dynamic_prompt = {
                    "question": retrieved_question,
                    "prompt": history_train[retrieved_question]
                }
            else:
                tmp_answer = results['metadatas'][0][idx]['answer']
                answer = tmp_answer.split("|")

                result_list, query_list, prompt_list = sllm.kgqa.kgqa(args, retrieved_question, table_data, relations)
                
                for _idx, prompt in enumerate(prompt_list):
                    prediction, _ = get_selfconsistency_res([result_list[_idx]])
                    if(evaluate_metaqa(prediction, answer)):
                        print(f"Retrieve for {retrieved_question} successfully")
                        temp_dynamic_prompt = {
                            "question":retrieved_question,
                            "prompt": prompt
                        }
                        history_train[retrieved_question] = prompt_list[_idx]
                        print(f"{retrieved_question}: {prompt_list[_idx]}")
                        break

            if temp_dynamic_prompt is not None:
                success_num += 1
                print(f"success_num: {success_num} retrieved_question:{retrieved_question}")
                dynamic_prompt.append(temp_dynamic_prompt)
            
            else:
                history_train[retrieved_question] = "None"

        if success_num == 0:
            print(f"For \"{question}\", only {success_num} prompts are successfully retrieved, less than {args.dynamically_prompt_num} prompts are retrieved.\nUse the naive prompt instead.\n")
            with open(args.prompt_path, 'r') as json_file:
                self.naive_prompt = json.load(json_file)
        elif success_num == args.dynamically_prompt_num:
            print(f"@@@@@@@@\"{question}\", {success_num} prompts are successfully retrieved.\n")
            self.naive_prompt = self.build_dynamic_prompt(args, dynamic_prompt)
        
        else:
            self.naive_prompt = self.build_dynamic_prompt(args, dynamic_prompt)
            with open(args.prompt_path, 'r') as json_file:
                static_prompt = json.load(json_file)
            
            print(f"@@@@@@@@\"{question}\", only {success_num} prompts are successfully retrieved, less than {args.dynamically_prompt_num} prompts are retrieved.\n")
            for i in range(args.dynamically_prompt_num - success_num):
                self.naive_prompt.append(static_prompt[-(i*2+2)])
                self.naive_prompt.append(static_prompt[-(i*2+1)])

        with open(args.retrieved_history, 'r+') as json_file:
            fcntl.flock(json_file, fcntl.LOCK_EX)

            existing_data = json.load(json_file)
            json_file.seek(0)

            for key in existing_data.keys():
                if key not in history_train.keys() or history_train[key] == "None":
                    history_train[key] = existing_data[key]

            json.dump(history_train, json_file, indent=4)
            json_file.truncate()
            fcntl.flock(json_file, fcntl.LOCK_UN)
    
    def build_dynamic_prompt(self, args, dynamic_prompt):
        prompt = []
        # get the instruction
        with open(args.prompt_path, 'r') as json_file:
            prompt.append(json.load(json_file)[0])
        
        for item in dynamic_prompt:
            # query = item['question']
            prompt.append(
                {
                    "role": "user",
                    "content": "Question: " + item['question']
                }
            )
            prompt.append(
                {
                    "role": "assistant", 
                    "content": item['prompt']
                }
            )

        return prompt
    
class wqsp_query_prompt():
    def __init__(self, args, DataFormat, table_data, collection = None):

        if collection is not None:
            self.retrieve_dynamic_prompt(args, DataFormat, table_data, collection)
        else:
            with open(args.prompt_path, 'r') as json_file:
                self.naive_prompt = json.load(json_file)

        prompt = self.wqsp_query_Prompt(DataFormat.question, DataFormat.TopicEntityName, DataFormat.First_step, DataFormat.Second_step)
        self.naive_prompt.append(
            {
                "role": "user",
                "content": prompt
            }
        )
    
    def retrieve_dynamic_prompt(self, args, DataFormat, table_data, collection):
        question = DataFormat.question

        dynamic_prompt = []
        success_num = 0
        
        history_train = {}
        if os.path.exists(args.retrieved_history) and collection is not None:
            with open(args.retrieved_history, 'r') as json_file:
                history_train = json.load(json_file)

        # results = collection.query(query_texts=[question], n_results=15)
        results = Get_Sampling_Results(args, question, collection)
        # results = collection.query(query_texts=[question], n_results=collection.count())
        assert results['metadatas'][0][0]['source'] != 'test', "Dynamic prompt cannot be generated using the test set"

        for idx in range(len(results['documents'][0])):
            print(f"Question: {question}.\nRetrieve for {idx}")
            if(success_num == args.dynamically_prompt_num): break

            temp_dynamic_prompt = None
            retrieved_question = results['documents'][0][idx]
            retrieved_First_step = json.loads(results['metadatas'][0][idx]['First_step'])
            retrieved_Second_step = json.loads(results['metadatas'][0][idx]['Second_step'])
            retrieved_TopicEntityName = results['metadatas'][0][idx]['TopicEntityName']
            retrieved_TopicEntityID = results['metadatas'][0][idx]['TopicEntityID']

            retrieved_tabledata = args.table_data[results['metadatas'][0][idx]['table']]

            if retrieved_question in history_train.keys():
                if(history_train[retrieved_question]=="None"): 
                    continue

                print(f"[{retrieved_question}] Retrieve from history successfully")
                temp_dynamic_prompt = {
                    "question": retrieved_question,
                    "prompt": history_train[retrieved_question],
                    "TopicEntityName": retrieved_TopicEntityName,
                    "First_step": retrieved_First_step,
                    "Second_step": retrieved_Second_step,
                }
            else:
                tmp_answer = results['metadatas'][0][idx]['answer']
                answer = tmp_answer.split("|")
                # try:
                result_list, _, prompt_list = \
                    sllm.wqspqa.wqspqa(
                        args, 
                        trans2CGdata.WQSPDataFormat(
                            retrieved_question,
                            retrieved_TopicEntityName,
                            retrieved_First_step,
                            retrieved_Second_step,
                            retrieved_TopicEntityID
                        ),
                        retrieved_tabledata
                    )
                for _idx, prompt in enumerate(prompt_list):
                    prediction, _ = get_selfconsistency_res([result_list[_idx]])
                    if(Wqsp.evaluate_wqsp(prediction, answer)):
                        print(f"Retrieve for {retrieved_question} successfully")
                        temp_dynamic_prompt = {
                            "question":retrieved_question,
                            "prompt": prompt,
                            "TopicEntityName": retrieved_TopicEntityName,
                            "First_step": retrieved_First_step,
                            "Second_step": retrieved_Second_step,
                        }
                        history_train[retrieved_question] = prompt_list[_idx]
                        print(f"{retrieved_question}: {prompt_list[_idx]}")
                        break

                # except Exception as e:
                #     print(f"Error in {retrieved_question}: {e}")
                #     continue

            if temp_dynamic_prompt is not None:
                success_num += 1
                print(f"success_num: {success_num} retrieved_question:{retrieved_question}")
                dynamic_prompt.append(temp_dynamic_prompt)
                
            else:
                history_train[retrieved_question] = "None"
                # with open(args.retrieved_history, 'w') as json_file: # save the error retrieved_pair
                #     json.dump(history_train, json_file, indent=4)

        if success_num == 0:
            print(f"For \"{question}\", only {success_num} prompts are successfully retrieved, less than {args.dynamically_prompt_num} prompts are retrieved.\nUse the naive prompt instead.\n")
            with open(args.prompt_path, 'r') as json_file:
                self.naive_prompt = json.load(json_file)
        elif success_num == args.dynamically_prompt_num:
            print(f"@@@@@@@@\"{question}\", {success_num} prompts are successfully retrieved.\n")
            self.naive_prompt = self.build_dynamic_prompt(args, dynamic_prompt)
        
        else:
            self.naive_prompt = self.build_dynamic_prompt(args, dynamic_prompt)
            with open(args.prompt_path, 'r') as json_file:
                static_prompt = json.load(json_file)
            
            print(f"@@@@@@@@\"{question}\", only {success_num} prompts are successfully retrieved, less than {args.dynamically_prompt_num} prompts are retrieved.\n")
            for i in range(args.dynamically_prompt_num - success_num):
                self.naive_prompt.append(static_prompt[-(i*2+2)])
                self.naive_prompt.append(static_prompt[-(i*2+1)])
                
        with open(args.retrieved_history, 'r+') as json_file:
            fcntl.flock(json_file, fcntl.LOCK_EX)

            existing_data = json.load(json_file)
            json_file.seek(0) 

            for key in existing_data.keys():
                if key not in history_train.keys() or history_train[key] == "None":
                    history_train[key] = existing_data[key]

            json.dump(history_train, json_file, indent=4)
            json_file.truncate()
            fcntl.flock(json_file, fcntl.LOCK_UN)

    def build_dynamic_prompt(self, args, dynamic_prompt):
        prompt = []
        # get the instruction
        with open(args.prompt_path, 'r') as json_file:
            prompt.append(json.load(json_file)[0])
        
        for item in dynamic_prompt:
            prompt.append(
                {
                    "role": "user",
                    "content": self.wqsp_query_Prompt(item['question'], item['TopicEntityName'], item['First_step'], item['Second_step'])
                }
            )
            prompt.append(
                {
                    "role": "assistant", 
                    "content": item['prompt']
                }
            )

        return prompt
    
    def wqsp_query_Prompt(self, question, TopicEntityName, First_step=None, Second_step=None):
        prompt = f"Question: {question}\nTopicEntityName: {TopicEntityName}"
        if First_step:
            if type(First_step) == dict:
                relation_string = str()
                for key in First_step.keys():
                    item = First_step[key]
                    relation_string += f"{item}|"
                relation_string = relation_string[:-1]
                
                prompt += f"\nFirst_step: {relation_string}"
            else:
                prompt += f"\nFirst_step: {First_step}"
        
        if Second_step:
            if type(Second_step) == dict:
                relation_string = str()
                for key in Second_step.keys():
                    item = Second_step[key]
                    relation_string += f"{item}|"
                relation_string = relation_string[:-1]
                prompt += f"\nSecond_step: {relation_string}"
            else:
                prompt += f"\nSecond_step: {Second_step}"

        return prompt


class temp_query_prompt():
    def __init__(self, args, DataFormat, table_data, collection = None):
        self.question = DataFormat.question
        self.model = args.model
        
        self.relation_list = DataFormat.relation_list
        self.annotation = DataFormat.annotation
        # self.prompt_path = args.prompt_path

        if collection is not None:
            self.retrieve_dynamic_prompt(args, DataFormat, table_data, collection)
        else:
            with open(args.prompt_path, 'r') as json_file:
                self.naive_prompt = json.load(json_file)

        prompt = self.temp_schema_Prompt(self.question, self.relation_list, self.annotation)
        self.naive_prompt.append(
            {
                "role": "user",
                "content": prompt
            }
        )
    
    def retrieve_dynamic_prompt(self, args, DataFormat, table_data, collection):
        question = DataFormat.question

        dynamic_prompt = []
        success_num = 0
        
        history_train = {}
        if os.path.exists(args.retrieved_history) and collection is not None:
            with open(args.retrieved_history, 'r') as json_file:
                history_train = json.load(json_file)

        results = collection.query(
            query_texts=[question], 
            n_results=15,
            where= {
                "$and":[
                    {
                        "type":{"$eq": DataFormat._type}
                    }, 
                    {
                        "answer_type":{"$eq": DataFormat.answer_type}
                    }
                ]
            }
        )
        # results = Get_Sampling_Results(args, question, collection)
        # results = collection.query(query_texts=[question], n_results=collection.count())
        assert results['metadatas'][0][0]['source'] != 'test', "Dynamic prompt cannot be generated using the test set"

        for idx in range(len(results['documents'][0])):
            print(f"Question: {question}.\nRetrieve for {idx}")
            if(success_num == args.dynamically_prompt_num): break

            temp_dynamic_prompt = None
            retrieved_question = results['documents'][0][idx]
            retrieved_relation_list = args.train_data[retrieved_question]['relation_list']
            retrieved_annotation = args.train_data[retrieved_question]['annotation']

            if retrieved_question in history_train.keys():
                if(history_train[retrieved_question]=="None"): 
                    continue

                print(f"[{retrieved_question}] Retrieve from history successfully")
                temp_dynamic_prompt = {
                    "question": retrieved_question,
                    "prompt": history_train[retrieved_question],
                    "relation_list": retrieved_relation_list,
                    "annotation": retrieved_annotation
                }
            else:
                tmp_answer = results['metadatas'][0][idx]['answer']
                answer = tmp_answer.split("|")
                # try:
                result_list, _, prompt_list = \
                    sllm.tempqa.tempqa(
                        args, 
                        trans2CGdata.CronQuestionDataFormat(
                            retrieved_question, 
                            retrieved_relation_list,
                            retrieved_annotation
                        ), 
                        table_data
                    )
                for _idx, prompt in enumerate(prompt_list):
                    prediction = Cron.get_selfconsistency_res([result_list[_idx]])
                    if(Cron.evaluate_cron(prediction, answer)):
                        print(f"Retrieve for {retrieved_question} successfully")
                        temp_dynamic_prompt = {
                            "question":retrieved_question,
                            "prompt": prompt,
                            "relation_list": retrieved_relation_list,
                            "annotation": retrieved_annotation
                        }
                        history_train[retrieved_question] = prompt_list[_idx]
                        print(f"{retrieved_question}: {prompt_list[_idx]}")
                        break

                # except Exception as e:
                #     print(f"Error in {retrieved_question}: {e}")
                #     continue

            if temp_dynamic_prompt is not None:
                success_num += 1
                print(f"success_num: {success_num} retrieved_question:{retrieved_question}")
                dynamic_prompt.append(temp_dynamic_prompt)
                
            else:
                history_train[retrieved_question] = "None"

        if success_num == 0:
            print(f"For \"{question}\", only {success_num} prompts are successfully retrieved, less than {args.dynamically_prompt_num} prompts are retrieved.\nUse the naive prompt instead.\n")
            with open(args.prompt_path, 'r') as json_file:
                self.naive_prompt = json.load(json_file)
        elif success_num == args.dynamically_prompt_num:
            print(f"@@@@@@@@\"{question}\", {success_num} prompts are successfully retrieved.\n")
            self.naive_prompt = self.build_dynamic_prompt(args, dynamic_prompt)
        
        else:
            self.naive_prompt = self.build_dynamic_prompt(args, dynamic_prompt)
            with open(args.prompt_path, 'r') as json_file:
                static_prompt = json.load(json_file)
            
            print(f"@@@@@@@@\"{question}\", only {success_num} prompts are successfully retrieved, less than {args.dynamically_prompt_num} prompts are retrieved.\n")
            for i in range(args.dynamically_prompt_num - success_num):
                self.naive_prompt.append(static_prompt[-(i*2+2)])
                self.naive_prompt.append(static_prompt[-(i*2+1)])
                
        with open(args.retrieved_history, 'r+') as json_file:
            fcntl.flock(json_file, fcntl.LOCK_EX) 

            existing_data = json.load(json_file)
            json_file.seek(0)

            for key in existing_data.keys():
                if key not in history_train.keys() or history_train[key] == "None":
                    history_train[key] = existing_data[key]

            json.dump(history_train, json_file, indent=4)
            json_file.truncate()
            fcntl.flock(json_file, fcntl.LOCK_UN)

    def build_dynamic_prompt(self, args, dynamic_prompt):
        prompt = []
        # get the instruction
        with open(args.prompt_path, 'r') as json_file:
            prompt.append(json.load(json_file)[0])

        for item in dynamic_prompt:
            prompt.append(
                {
                    "role": "user",
                    "content": self.temp_schema_Prompt(item['question'], item['relation_list'], item['annotation'])
                }
            )
            prompt.append(
                {
                    "role": "assistant", 
                    "content": item['prompt']
                }
            )

        return prompt
    
    
    def temp_schema_Prompt(self, question, relation_list, annotation):
        prompt = f"Question: {question}\nRelations: {relation_list}\nannotation: {annotation}"
        return prompt

class retrieve_prompt():
    def __init__(self, head_entity, retrieve, CG_relations):
        self.head_entity = head_entity
        self.retrieve = retrieve
        self.CG_relations = CG_relations
        if bool(re.compile(r'[\u4e00-\u9fa5]').search(head_entity)): #换中文prompt
            self.naive_prompt = f"给定一个实体: “{head_entity}”，以及可能的实体类别: {CG_relations}. 请选择“{head_entity}”最可能属于的类别名称："
        else:
            self.naive_prompt = [
              {
                "role": "user",
                # "content": f"Given an:\"{head_entity}\", and the result of this entity retrieve: {retrieve}, and all candidate: {CG_relations}. Please select {head_entity} is most likely to belong to the category of the name:"
                # "content": f"Given an:\"{head_entity}\" and all candidate: {CG_relations}. Please select {head_entity} is most likely to belong to the category of the name:"
                "content": f"Please provide the correct relation type for \"{head_entity}\" and the options are: {CG_relations}. The relation type:"
              }
            ]