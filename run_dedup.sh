python src/minhash_deduplication.py --data-dir stage1_dataset_nj.json \
    --split train \
    --column content \
    --cache-dir .cache \
    --min-ngram-size 5 \
    --output "dataset_stage1_dedup.jsonl"