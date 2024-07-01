#!/usr/bin/env python
# coding=utf-8

"""
data: 5-23-2024 13:29
author: y31506
"""

import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import time
import re
import json


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_step1_path", help="Path to the step1 dataset", default="Cmw_Dataset_Step1_test.json")
    parser.add_argument("--fim_rate", help="The rate of FIM", default=0.5)
    parser.add_argument("--fim_spm_rate", help="The rate of SPM", default=0.5)
    parser.add_argument("--seq_length", help="The sequence length", default=8192)
    parser.add_argument("--seed", help="seed number", default=42)
    parser.add_argument("--tokenizer_path", help="The path to the tokenizer", default="codeqwen_tokenizer")
    parser.add_argument("--cmw_dataset_output_path", help="The path to the output dataset", default="fimdataset_test.json")

    return parser.parse_args()




class CmwDatasetFim:

    def __init__(self, dataset_step1_path, fim_rate, fim_spm_rate, seed,
                 tokenizer_path, seq_length, cmw_dataset_output_path):

        self.dataset_step1_path = dataset_step1_path
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seq_length = seq_length
        self.seed = seed
        self.tokenizer_path = tokenizer_path

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.eos_token = self.tokenizer.eos_token
        self.prefix_token = "<fim_prefix>"
        self.suffix_token = "<fim_suffix>"
        self.middle_token = "<fim_middle>"
        self.repo_token = "<reponame>"
        self.file_token = "<file_sep>"
        self.max_span_len = 100  # fim_middle的最大长度
        self.min_span_len = 3  # fim_middle的最小长度
        self.min_file_len = 150  # 进行FIM处理的最小文件长度
        self.np_rng = np.random.RandomState(seed=seed)  # rng state for FIM
        self.cmw_dataset_output_path = cmw_dataset_output_path

    def get_str_tok_len(self, string):
        return len(self.tokenizer.encode(string))

    def fim_permute_sequence(self, sequence):
        return permute(
            sequence,
            np_rng=self.np_rng,
            fim_spm_rate=self.fim_spm_rate,
            suffix_tok=self.suffix_token,
            prefix_tok=self.prefix_token,
            middle_tok=self.middle_token,
            eos_tok=self.eos_token,
            max_span_len=self.max_span_len,
            min_span_len=self.min_span_len
        )

    def process_fim_and_repo(self, cur_seq, cur_seq_tok_len, cur_seq_file_name):
        # 进行FIM处理,token数小于150的文件不进行FIM处理
        if cur_seq_tok_len >= self.min_file_len and self.np_rng.binomial(1, self.fim_rate):
            cur_seq = self.fim_permute_sequence(cur_seq)
        else:
            cur_seq = cur_seq + self.eos_token  # 注意不进行FIM处理的sequence要添加eos token
        # 添加文件名信息  <file_sep>filename{file_content}
        file_str = self.file_token + cur_seq_file_name + "\n"
        cur_seq = file_str + cur_seq
        cur_seq_tok_len = self.get_str_tok_len(cur_seq)

        return cur_seq, cur_seq_tok_len

    # charactor-level FIM
    def build_fim_dataset(self):
        # load the step1 dataset
        raw_dataset = get_json_data(self.dataset_step1_path)
        # build the FIM dataset
        fim_dataset = []

        potential_extra_tok_num = 55  # 潜在的repo信息以及fim符号+截断造成的编码膨胀token数

        # 按照最深一级目录进行处理fim数据集(伪repo级别)
        for repo, repo_items in tqdm(raw_dataset.items()):
            # 核心处理逻辑，涉及到拼接与截断
            cur_data_token_count = 0
            cur_data_str = ""
            repo_str = self.repo_token + repo
            repo_tok_len = self.get_str_tok_len(repo_str)
            # 直接遍历repo_items将会无法处理截断后的剩余部分，因此需要在遍历过程中维护一个待处理的列表
            items_to_process = repo_items.copy()

            while items_to_process:
                item = items_to_process.pop(0)
                cur_seq = item["content"]
                cur_seq_tok_len = item["token_num"]
                cur_seq_file_name = item["file_name"]

                if cur_data_token_count + cur_seq_tok_len + potential_extra_tok_num <= self.seq_length:
                    cur_seq_processed, cur_seq_tok_len_processed = self.process_fim_and_repo(cur_seq, cur_seq_tok_len, cur_seq_file_name)
                    cur_data_str += cur_seq_processed
                    cur_data_token_count += cur_seq_tok_len_processed
                    # 如果累积到足够的长度，就将当前数据添加到数据集中
                    if cur_data_token_count >= 0.9 * self.seq_length:
                        # seq: '<reponame>reponame\n<file_sep>filename\n' + content
                        cur_data_str = repo_str + "\n" + cur_data_str
                        cur_data_token_count += repo_tok_len + 1
                        fim_dataset.append({"content": cur_data_str, "token_num": cur_data_token_count})
                        cur_data_str = ""
                        cur_data_token_count = 0

                else:
                    # 当前序列需要截断
                    # 注意这里的split计算需要根据实际情况调整
                    split_index = self.seq_length - cur_data_token_count - potential_extra_tok_num
                    if split_index < cur_seq_tok_len:
                        # 根据split_index截断序列
                        cur_seq_tok_id = self.tokenizer.encode(cur_seq, add_special_tokens=False)
                        split_seq_tok_id = cur_seq_tok_id[:split_index]
                        remaining_seq_tok_id = cur_seq_tok_id[split_index:]
                        split_seq = self.tokenizer.decode(split_seq_tok_id, skip_special_tokens=True)
                        remaining_seq = self.tokenizer.decode(remaining_seq_tok_id, skip_special_tokens=True)
                        split_seq_tok_len = len(split_seq_tok_id)
                        remaining_seq_tok_len = len(remaining_seq_tok_id)

                        # 将截断后的序列部分进行处理并添加到cur_data_str中
                        split_seq_processed, split_seq_tok_len_processed = self.process_fim_and_repo(split_seq, split_seq_tok_len, cur_seq_file_name)
                        cur_data_str += split_seq_processed
                        cur_data_token_count += split_seq_tok_len_processed

                        # 如果累积到足够的长度，就将当前数据添加到数据集中
                        if cur_data_token_count >= 0.9 * self.seq_length:
                            cur_data_str = repo_str + "\n" + cur_data_str
                            cur_data_token_count += repo_tok_len + 1
                            fim_dataset.append({"content": cur_data_str, "token_num": cur_data_token_count})
                            cur_data_str = ""
                            cur_data_token_count = 0

                            # 将剩余部分作为下一个item重新加入列表
                            items_to_process.insert(0, {'content': remaining_seq, 'token_num': remaining_seq_tok_len, "file_name": cur_seq_file_name})

                        else:
                            print("如果执行到了这里，说明有bug")

                # 如果items_to_process为空，但是cur_data_str不为空，需要将cur_data_str添加到fim_dataset中
                if not items_to_process and cur_data_str:
                    cur_data_str = repo_str + "\n" + cur_data_str
                    cur_data_token_count += repo_tok_len + 1
                    fim_dataset.append({"content": cur_data_str, "token_num": cur_data_token_count})

        # 将fim_dataset保存到文件
        with open(self.cmw_dataset_output_path, "w", encoding="utf-8") as f:
            for item in fim_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def permute(sample, np_rng, fim_spm_rate, suffix_tok=None,
            prefix_tok=None, middle_tok=None, eos_tok=None,
            max_span_len=256, min_span_len=3):
    """
    Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it.
    Maintain the same sample length (if transform creates a few extra tokens, drop them).
    """

    while True:
        # generate a number between max_span_len and min_span_len
        middle_length = np_rng.randint(min_span_len, max_span_len+1)

        # make sure the middle length is not longer than sample length
        if len(sample) - middle_length > 1:
            break

    # randomly select the length of the prefix length
    prefix_length = np_rng.randint(0, len(sample) - middle_length)

    prefix = sample[:prefix_length]
    middle = sample[prefix_length:prefix_length + middle_length]
    suffix = sample[prefix_length + middle_length:]

    if np_rng.binomial(1, fim_spm_rate):
        # SPM (variant 2 from FIM paper)
        new_sample = prefix_tok + suffix_tok + suffix + middle_tok + prefix + middle + eos_tok

    else:
        # PSM
        new_sample = prefix_tok + prefix + suffix_tok + suffix + middle_tok + middle + eos_tok

    return new_sample


def permute_no_limit(sample, np_rng, fim_rate, fim_spm_rate, suffix_tok=None,
            prefix_tok=None, middle_tok=None, eos_tok=None,
            max_span_len=256, min_span_len=3, tokenizer=None):
    """
    Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it.
    Maintain the same sample length (if transform creates a few extra tokens, drop them).
    """

    try:
        # A boundary can be =0 (prefix will be empty)
        # a boundary can be =len(contents) (suffix will be empty)
        # The two boundaries can be equal (middle will be empty)
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()
    except ValueError as e:
        print(len(sample), sample)
        print(e)
        raise e

    prefix = sample[:boundaries[0]]
    middle = sample[boundaries[0]:boundaries[1]]
    suffix = sample[boundaries[1]:]

    if np_rng.binomial(1, fim_spm_rate):
        # SPM (variant 2 from FIM paper)
        new_sample = prefix_tok + suffix_tok + suffix + middle_tok + prefix + middle + eos_tok

    else:
        # PSM
        new_sample = prefix_tok + prefix + suffix_tok + suffix + middle_tok + middle + eos_tok

    return new_sample


if __name__ == "__main__":
    args = arg_parser()
    start_time = time.time()
    cmw_dataset_fim = CmwDatasetFim(args.dataset_step1_path, args.fim_rate, args.fim_spm_rate, args.seed,
                                    args.tokenizer_path, args.seq_length, args.cmw_dataset_output_path)
    cmw_dataset_fim.build_fim_dataset()
    print("FIM dataset has been successfully built")
    print(f"Total time: {time.time() - start_time}")



