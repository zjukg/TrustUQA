OUTPUTPAT_MAIN=output/wqsp
OUTPUTPATH_NAME=$OUTPUTPAT_MAIN/name
mkdir -p $OUTPUTPATH_NAME
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python CGdata_for_WQSP.py \
--key api_key.txt --num_process 10 \
--openai_url https://api.openai.com/v1/ \
--folder_path dataset/WQSP/data/test \
--data_path dataset/WQSP/question/test/name_question.jsonl \
--prompt_path structllm/prompt_/WQSP/WQSP_name.json \
--SC_Num 5 \
--model gpt-3.5-turbo-0613 \
--output_detail_path $OUTPUTPATH_NAME/output_detail \
--output_result_path $OUTPUTPATH_NAME/output_result \
--debug 0 \
--retriever SentenceBERT \
--retriever_align SentenceBERT \
--dynamically_prompt_num 8 \
--retrieved_history structllm/prompt_/WQSP/wqsp_name_train_candidate_demo.json \
--sampling TopK \
--sampling_num 15

cat $OUTPUTPATH_NAME/output_detail* > $OUTPUTPATH_NAME/all_detail.txt
cat $OUTPUTPATH_NAME/output_result* > $OUTPUTPATH_NAME/all_result.txt

OUTPUTPATH_UNNAME=output/wqsp/unname
mkdir -p $OUTPUTPATH_UNNAME
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python CGdata_for_WQSP.py \
--key api_key.txt --num_process 10 \
--openai_url https://api.openai.com/v1/ \
--folder_path dataset/WQSP/data/test \
--data_path dataset/WQSP/question/test/unname_question.jsonl \
--prompt_path structllm/prompt_/WQSP/WQSP_unname.json \
--SC_Num 5 \
--model gpt-3.5-turbo-0613 \
--output_detail_path $OUTPUTPATH_UNNAME/output_detail \
--output_result_path $OUTPUTPATH_UNNAME/output_result \
--debug 0 \
--retriever SentenceBERT \
--retriever_align SentenceBERT \
--dynamically_prompt_num 8 \
--retrieved_history structllm/prompt_/WQSP/wqsp_unname_train_candidate_demo.json \
--sampling TopK \
--sampling_num 15

cat $OUTPUTPATH_UNNAME/output_detail* > $OUTPUTPATH_UNNAME/all_detail.txt
cat $OUTPUTPATH_UNNAME/output_result* > $OUTPUTPATH_UNNAME/all_result.txt

cat $OUTPUTPAT_MAIN/*/all_result.txt > $OUTPUTPAT_MAIN/all_result.txt

python evaluate/evaluate_for_wqsp.py \
--ori_path $OUTPUTPAT_MAIN/all_result.txt \
--error_cases_output $OUTPUTPAT_MAIN/error_cases.txt \
--write_flag True

