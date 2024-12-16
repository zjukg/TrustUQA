import os
import json
import chromadb
import torch
import openai
import time
import re
from torch import Tensor
from openai import OpenAI
from tqdm.autonotebook import trange
from typing import List, Union, TypeVar, Dict
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Images
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection
from tenacity import retry, wait_random_exponential, stop_after_attempt

Embeddable = Union[Documents, Images]
D = TypeVar("D", bound=Embeddable, contravariant=True)

@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(6))
def get_embedding_openai(text, model="text-embedding-ada-002") -> List[float]:
    if isinstance(text, str):
        max_retries = 20
        for i in range(max_retries):
            try:
                
                client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
                result = client.embeddings.create(input=[text], model=model).data[0].embedding
                
                break
            except openai.APIConnectionError as e:
                
                if i < max_retries - 1:  
                    time.sleep(1)  
                    continue
                else:  
                    raise
        return result
    elif isinstance(text, list):
        max_retries = 20
        for i in range(max_retries):
            try:
                
                result = client.embeddings.create(input=text, model=model)
                
                break
            except openai.APIConnectionError as e:
                
                if i < max_retries - 1:  
                    time.sleep(1)  
                    continue
                else:  
                    raise
        return result

class NewEmbeddingFunction(EmbeddingFunction):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder

    def __call__(self, input: D) -> Embeddings:
        embeddings = self.encoder.encode(input)
        return embeddings


class EncoderAda002:
    def encode(
        self,
        text: List[str],
        batch_size: int = 16,
        show_progress_bar: bool = False,
        **kwargs
    ) -> List[Tensor]:
        text_embeddings = []
        for batch_start in trange(
            0, len(text), batch_size, disable=not show_progress_bar
        ):
            batch_end = batch_start + batch_size
            batch_text = text[batch_start:batch_end]
            
            assert "" not in batch_text
            resp = get_embedding_openai(batch_text)
            for i, be in enumerate(resp.data):
                assert (
                    i == be.index
                )  
            batch_text_embeddings = [e.embedding for e in resp.data]
            text_embeddings.extend(batch_text_embeddings)

        return text_embeddings


class OpenaiAda002:
    def __init__(self) -> None:
        self.q_model = EncoderAda002()
        self.doc_model = self.q_model

    def encode(
        self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> List[Tensor]:
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs)


class Encoder:

    def __init__(self, encoder_name: str) -> None:
        self.encoder_name = encoder_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        
        
        if encoder_name == "text-embedding-ada-002":
            self.encoder = OpenaiAda002()
            self.ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ["OPENAI_API_KEY"],
                model_name="text-embedding-ada-002",
            )
        elif encoder_name == "SentenceBERT":
            path = 'msmarco-distilbert-base-tas-b'
            self.encoder = SentenceTransformer(
                path, device=self.device
            )
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=path
            )
        elif encoder_name == "ANCE":
            path = "msmarco-roberta-base-ance-firstp"
            self.encoder = SentenceTransformer(
                path, device=self.device
            )
            self.ef = NewEmbeddingFunction(self.encoder)
        elif encoder_name == "DPR":
            path = 'facebook-dpr-question_encoder-single-nq-base'
            self.encoder = SentenceTransformer(
                path,
                device=self.device,
            )
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=path
            )


def _get_embedding_and_save_to_chroma(
    data: List[Dict[str, str]],
    collection: Collection,
    encoder: Encoder,
    batch_size: int = 64,
):
    encoder_ = encoder.encoder

    docs = [item["question"] for item in data]
    meta_keys = list(data[0].keys())
    del meta_keys[meta_keys.index("question")]

    embeddings = encoder_.encode(docs, batch_size=batch_size, show_progress_bar=True)
    
    
    
    
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()
    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
            collection.add(
                embeddings=embeddings[i : i + 20000],
                documents=docs[i : i + 20000],
                metadatas=[
                    {key: data[i][key] for key in meta_keys}
                    for i in range(i, min(len(embeddings), i + 20000))
                ],
                ids=[str(i) for i in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[
                {key: data[i][key] for key in meta_keys} for i in range(len(docs))
            ],
            ids=[str(i) for i in range(len(docs))],
        )
    return collection


def get_embedding(dataset_path: str, retriever: str, chroma_dir: str):
    dataset_name = ""
    if "metaqa" in dataset_path.lower():
        dataset_name = "MetaQA"
    elif "wtq" in dataset_path.lower():
        dataset_name = "WTQ"
    elif "wikisql" in dataset_path.lower():
        dataset_name = "WikiSQL"
    elif "cronquestion" in dataset_path.lower():
        dataset_name = "CronQuestion"
    elif "wqsp" in dataset_path.lower():
        dataset_name = "WQSP"
    else:
        pass
    chroma_path = os.path.join(chroma_dir, retriever, dataset_name)

    encoder = Encoder(retriever)
    
    if dataset_name == 'MetaQA':
        name = (
            "1-hop"
            if "1-hop" in dataset_path
            else ("2-hop" if "2-hop" in dataset_path else "3-hop")
        )
    elif dataset_name == 'WQSP':
        name = (
            "unname"
            if "unname" in dataset_path
            else "name"
        )
    else:
        name = "main"
    
    embedding_function = encoder.ef
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
    if not collection.count():
        if dataset_name == "MetaQA":
            with open(dataset_path.replace("test", "train"), "r") as f:
                data_train = json.loads(f.read())

            with open(dataset_path.replace("test", "dev"), "r") as f:
                data_valid = json.loads(f.read())

            template_cnt = dict()
            KGQA_data = []
            for KG_id in data_train.keys():
                question = data_train[KG_id]["question"]
                answer = data_train[KG_id]["answer"]
                template = re.sub(r'\[.*?\]', '[]', question)
                if(template not in template_cnt):
                    template_cnt[template] = 0
                else:
                    if(template_cnt[template]>=50): continue
                    template_cnt[template] += 1
                item = {
                    "question": question,
                    "answer": "|".join(answer),
                    "source": "train",
                    "template": template
                }
                KGQA_data.append(item)

            for KG_id in data_valid.keys():
                question = data_valid[KG_id]["question"]
                answer = data_valid[KG_id]["answer"]
                template = re.sub(r'\[.*?\]', '[]', question)
                if(template not in template_cnt):
                    template_cnt[template] = 0
                else:
                    if(template_cnt[template]>=50): continue
                    template_cnt[template] += 1
                item = {
                    "question": question,
                    "answer": "|".join(answer),
                    "source": "valid",
                    "template": template
                }
                KGQA_data.append(item)

            _get_embedding_and_save_to_chroma(KGQA_data, collection, encoder)

        elif dataset_name == "WTQ":
            with open(dataset_path.replace("test", "train"), "r") as f:
                data_train = json.loads(f.read())

            TableQA_data = []
            for k, v in data_train.items():
                for qa in v:
                    question = qa[0]
                    answer = qa[1]
                    item = {
                        "question": question,
                        "answer": "|".join(answer),
                        "source": "train",
                        "table": k.split(".")[0],
                    }
                    TableQA_data.append(item)
            _get_embedding_and_save_to_chroma(TableQA_data, collection, encoder)
        
        elif dataset_name == "WikiSQL":
            with open(dataset_path.replace("test", "dev"), "r") as f:
                data_dev = json.loads(f.read())
            
            with open(dataset_path.replace("test", "train"), "r") as f:
                data_train = json.loads(f.read())
            
            TableQA_data = []
            for k, v in data_dev.items():
                for qa in v:
                    question = qa[0]
                    answer = qa[1]
                    item = {
                        "question": question,
                        "answer": "|".join(str(a) for a in answer),
                        "source": "dev",
                        "table": k.split(".")[0],
                    }
                    TableQA_data.append(item)

            for k, v in data_train.items():
                for qa in v:
                    question = qa[0]
                    answer = qa[1]
                    item = {
                        "question": question,
                        "answer": "|".join(str(a) for a in answer),
                        "source": "train",
                        "table": k.split(".")[0],
                    }
                    TableQA_data.append(item)
    
            _get_embedding_and_save_to_chroma(TableQA_data, collection, encoder)
        
        elif dataset_name == "CronQuestion":
            with open(dataset_path.replace("test", "valid"), "r") as f:
                data_dev = json.loads(f.read())
            
            TableQA_data = []
            for k, v in data_dev.items():
                question = v['question']

                annotation = v['annotation']
                id2entity = v['entities']
                resEntity = [item for entity_key, item in id2entity.items() if entity_key not in annotation.values() ]
                if resEntity != []:
                    
                    if len(resEntity) > 1 or '{tail2}' not in question:
                        assert False, (resEntity, question)

                    
                    question = question.replace('{tail2}', resEntity[0])

                answer = v['answer'][1]
                item = {
                    "question": question,
                    "answer": "|".join(str(a) for a in answer),
                    "source": "dev",
                    "type": v['type'],
                    "answer_type": v['answer_type']
                    
                }
                
                TableQA_data.append(item)
            _get_embedding_and_save_to_chroma(TableQA_data, collection, encoder)

        elif dataset_name == "WQSP":

            WQSP_data = []
            with open(dataset_path.replace("test", "train"), "r") as f_train:
                for line in f_train:
                    obj = json.loads(line)
                    question = obj['question']
                    answers = obj['answers']
                    table = obj['ID']
                    First_step = json.dumps(obj['First_step'])
                    Second_step = json.dumps(obj['Second_step'])
                    TopicEntityName = obj['entities'][0][0]
                    TopicEntityID = obj['entities'][0][1]
                    answer = [ item[0][3:] if item[0][:3] == "ns:" else item[0] for item in answers ]
                    item = {
                        "question": question,
                        "answer": "|".join(answer),
                        "source": "train",
                        "table": table,
                        "First_step": First_step,
                        "Second_step": Second_step,
                        "TopicEntityName": TopicEntityName,
                        "TopicEntityID": TopicEntityID
                    }
                    WQSP_data.append(item)

            with open(dataset_path.replace("test", "dev"), "r") as f_dev:
                for line in f_dev:
                    obj = json.loads(line)
                    question = obj['question']
                    answers = obj['answers']
                    table = obj['ID']
                    First_step = json.dumps(obj['First_step'])
                    Second_step = json.dumps(obj['Second_step'])
                    TopicEntityName = obj['entities'][0][0]
                    TopicEntityID = obj['entities'][0][1]
                    answer = [ item[0][3:] if item[0][:3] == "ns:" else item[0] for item in answers ]
                    item = {
                        "question": question,
                        "answer": "|".join(answer),
                        "source": "dev",
                        "table": table,
                        "First_step": First_step,
                        "Second_step": Second_step,
                        "TopicEntityName": TopicEntityName,
                        "TopicEntityID": TopicEntityID
                    }
                    WQSP_data.append(item)
            
            _get_embedding_and_save_to_chroma(WQSP_data, collection, encoder)
            
        else:
            pass

    return collection