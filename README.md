# ChamaleonLLM
ChamaleonLLM: Batch-Aware Dynamic Low-Rank Adaptation via Inference-Time Clusters

# To Reproduce:
python main.py --lm_model_name "gpt2" --dataset_name "wikitext" --dataset_config "wikitext-2-raw-v1" --text_field "text" --batch_size 32 --max_length 256 --num_epochs 10 --rank 4 --alpha 1.0 --lr 5e-4
