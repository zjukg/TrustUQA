OUTPUTPATH=output/wtq
mkdir -p $OUTPUTPATH
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python CGdata_for_tableqa.py \
--key api_key.txt --num_process 10 \
--openai_url https://api.openai.com/v1/ \
--folder_path dataset/WTQ/csv \
--data_path dataset/WTQ/test.jsonl \
--prompt_path structllm/prompt_/WTQ.json \
--SC_Num 5 \
--model gpt-3.5-turbo-0613 \
--output_detail_path $OUTPUTPATH/output_detail \
--output_result_path $OUTPUTPATH/output_result \
--debug 0 \
--retriever SentenceBERT \
--retriever_align SentenceBERT \
--dynamically_prompt_num 8 \
--retrieved_history structllm/prompt_/wtq_train_candidate_demo.json \
--sampling TopK \
--sampling_num 15

cat $OUTPUTPATH/output_detail* > $OUTPUTPATH/all_detail.txt
cat $OUTPUTPATH/output_result* > $OUTPUTPATH/all_result.txt

python evaluate/evaluate_for_tableqa.py \
--ori_path $OUTPUTPATH/all_result.txt \
--error_cases_output $OUTPUTPATH/error_cases.txt \
--write_flag True