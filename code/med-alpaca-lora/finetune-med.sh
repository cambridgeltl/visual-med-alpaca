python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path '/path/to/med_alpaca_data_clean.json' \
    --micro_batch_size 32 \
    --output_dir './med-alpaca-lora' 
    