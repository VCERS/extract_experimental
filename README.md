# Introduction

this project implements experimental related content extraction algorithm

# Usage

## Install prerequisite

```shell
python3 -m pip install -r requirements.txt
```

## start text-generate-inference

```shell
model=Qwen/Qwen2.5-7B-Instruct
docker run --gpus 4,5,6,7 --shm-size 1g -p 8080:80 -v /home/xieyi/raid/huggingface:/data ghcr.io/huggingface/text-generation-inference --model-id $model --max-input-length 52207 --max-batch-prefill-tokens 52207 --max-total-tokens 131072 --max-batch-size 32 --num-shard 4
```

## run extraction

```shell
python3 main.py --input_dir <path/to/test> --output_dir <path/to/results>
```
