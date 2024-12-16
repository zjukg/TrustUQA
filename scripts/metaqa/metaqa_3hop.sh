OUTPUTPATH=output/metaqa/3hop
mkdir -p $OUTPUTPATH
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python CGdata_for_metaqa.py \
--key api_key.txt --num_process 10 \
--openai_url https://api.openai.com/v1/ \
--folder_path dataset/MetaQA_CG/kg.txt \
--data_path dataset/MetaQA_CG/test/3-hop_qa.jsonl \
--prompt_path structllm/prompt_/MetaQA/3hop.json \
--SC_Num 5 \
--model gpt-3.5-turbo-0613 \
--output_detail_path $OUTPUTPATH/output_detail \
--output_result_path $OUTPUTPATH/output_result \
--debug 0 \
--retriever SentenceBERT \
--retriever_align SentenceBERT \
--dynamically_prompt_num 8 \
--retrieved_history structllm/prompt_/MetaQA/train_candidate_demo_3hop.json \
--sampling TopK \
--sampling_num 15

cat $OUTPUTPATH/output_detail* > $OUTPUTPATH/all_detail.txt
cat $OUTPUTPATH/output_result* > $OUTPUTPATH/all_result.txt

python evaluate/evaluate_for_metaqa.py \
--ori_path $OUTPUTPATH/all_result.txt \
--error_cases_output $OUTPUTPATH/error_cases.txt \
--write_flag True