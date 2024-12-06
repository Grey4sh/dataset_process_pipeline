# dataset_process_pipeline

convert the source code repo to post-pretrain dataset for CodeQwen

## start

1.首先安装依赖

2.执行 pipeline.sh脚本 注意运行脚本前需要将该工程下的*run_dedup.sh*脚本的权限设为可执行（例如 sudo chmod 777 run_dedup.sh）

```shell
python src/pipeline.py \
--dir_name /young/v9 \
--tokenizer_path codeqwen_tokenizer \
--project_language "c" \
--bigfile_filter True \
--seq_length 4096 \
--cmw_dataset_output_path pipeline_v9trunk_test.json
```

参数解释：

- dir_name：需要训练的代码仓库绝对路径
- tokenizer_path：模型tokenizer路径（该工程已自带，不用动）
- project_language：代码语言（当前仅支持c和java）
- bigfile_filter：超大文件过滤，默认为True
- seq_length：处理完成后每条数据集的token长度，默认4096，不要动
- cmw_dataset_output_path：处理完成后数据集的名称

