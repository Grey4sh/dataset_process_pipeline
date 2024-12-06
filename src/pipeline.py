#!/usr/bin/env python
# coding=utf-8

"""
data: 5-21-2024 13:29
author: y31506
"""

import argparse
import chardet
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import time
import re
import subprocess
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# 要跳过的文件夹后缀列表
skip_suffixes = ["/grpcplugin", "/resource", "/webplugin", "/verifysuite"]


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_name", help="待处理项目路径", default=None)
    parser.add_argument("--bigfile_filter", help="是否对大文件进行剔除", default=False)
    parser.add_argument("--project_language", help="项目语言选择", default="c")
    parser.add_argument("--fim_rate", help="The rate of FIM", default=0.5)
    parser.add_argument("--fim_spm_rate", help="The rate of SPM", default=0.5)
    parser.add_argument("--seq_length", help="The sequence length", default=4096)
    parser.add_argument("--seed", help="seed number", default=42)
    parser.add_argument("--tokenizer_path", help="The path to the tokenizer", default="codeqwen_tokenizer")
    parser.add_argument("--cmw_dataset_output_path", help="The path to the output dataset",
                        default="fimdataset_test.json")
    return parser.parse_args()


# 去除Comware风格函数头，文件头以及修改信息注释
def remove_extra_comment(code_string: str):
    # Define regex patterns for multiline comments
    main_pattern = r'/\*{1,2}[\s\S]*?\*/'  # 匹配多行函数头&文件头注释
    pn_pattern = r'/\*.*?PN:.*?\*/'  # 匹配修改信息注释（主要针对sr88drv）

    # Find all matches of multiline comments
    main_matches = re.findall(main_pattern, code_string)
    pn_matches = re.findall(pn_pattern, code_string)

    # Remove matches that are less than 5 lines long
    for match in main_matches:
        if match.count('\n') >= 5:  # Check if the comment is 5 lines or longer
            code_string = code_string.replace(match, '')  # Remove the comment

    # Remove matches with PN pattern
    for match in pn_matches:
        code_string = code_string.replace(match, '')  # Remove the comment

    return code_string


def run_shell_script(script_path):
    print(f"Executing shell script: {script_path}")
    result = subprocess.run(script_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        print("Shell script executed successfully")
        print(result.stdout)
    else:
        print("Shell script execution failed")
        print(result.stderr)


def load_nested_dict(nested_dict_pth):
    # 读取JSON文件
    with open(nested_dict_pth, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def convert_nested_dict_to_normjson(nested_dict_pth):
    # # 读取JSON文件
    # with open(nested_dict_pth, 'r', encoding='utf-8') as file:
    #     data = json.load(file)
    data = nested_dict_pth
    # 步骤2：将字典的结构变为一个标准的jsonlist
    json_list = []

    for repo_name, files in data.items():
        for file in files:
            json_list.append({
                "repo_name": repo_name,
                "file_name": file["file_name"],
                "content": file["content"],
                "token_num": file["token_num"]
            })
    return json_list


def convert_norm_json_to_nested_dict(norm_json_pth):
    # 读取 JSON 列表文件
    with open(norm_json_pth, 'r', encoding='utf-8') as infile:
        json_list = json.load(infile)
    # 初始化一个默认字典，用于存储转换后的数据
    data = defaultdict(list)

    # 遍历 JSON 列表，将数据转换为嵌套字典格式
    for item in json_list:
        reponame = item.pop('repo_name')  # 移除并获取 reponame 字段
        data[reponame].append(item)

    # 将 defaultdict 转换为普通字典
    data = dict(data)
    return data


def get_json_data(file_path):
    if file_path.endswith("json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    else:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                item = json.loads(line)
                data.append(item)

    return data


def should_skip_dir(dir_path):
    """检查目录路径是否以任何一个指定的后缀结尾，如果是则返回 True"""
    for suffix in skip_suffixes:
        if dir_path.endswith(suffix):
            return True
    return False


class CmwDataset:
    def __init__(self, dir_name, tokenizer_path, bigfile_filter, language):
        self.dir_name = dir_name
        self.tokenizer_path = tokenizer_path
        self.bigfile_filter = bigfile_filter
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.unexpected_encoder_num = 0
        self.language = language

    def collect_files(self, repo_path):
        """收集repo中的.c和.h文件，并按要求排序；对于Java，直接返回所有文件"""
        all_files = []

        # 遍历目录
        for dirpath, dirnames, filenames in os.walk(repo_path):
            # 过滤出需要跳过的目录
            dirnames[:] = [d for d in dirnames if not should_skip_dir(os.path.join(dirpath, d))]

            for filename in filenames:
                if filename.endswith('.c') or filename.endswith('.h') or filename.endswith('.java'):
                    file_path = os.path.join(dirpath, filename)
                    all_files.append((filename, file_path))

        if 'c' == self.language:
            # 对C语言的文件进行排序（.c 和 .h 文件）
            def sort_files(file_list):
                file_dict = {}
                for filename, file_path in file_list:
                    base_name, ext = os.path.splitext(filename)
                    if base_name not in file_dict:
                        file_dict[base_name] = []
                    file_dict[base_name].append((filename, file_path))

                sorted_list = []
                for base_name in sorted(file_dict.keys()):
                    matched_files = sorted(file_dict[base_name], key=lambda x: x[0].endswith('.h'), reverse=True)
                    sorted_list.extend(matched_files)
                return sorted_list

            # 对C语言文件进行排序
            sorted_files = sort_files(all_files)

        elif 'java' == self.language:
            # 对Java语言，直接返回所有文件，不进行排序
            sorted_files = all_files

        else:
            # 如果不是C或JAVA，返回空列表或其他处理方式
            raise "only support 'c' or 'java'"

        return sorted_files

    def traverse_and_collect(self):
        root = self.dir_name
        repo_files = {}
        with ThreadPoolExecutor() as executor:
            future_to_repo = {}
            for component in tqdm(os.listdir(root)):
                component_path = os.path.join(root, component)
                if os.path.isdir(component_path):
                    # 直接处理最后一级目录，不再根据特定子目录区分
                    for dirpath, dirnames, filenames in os.walk(component_path):
                        dirnames[:] = [d for d in dirnames if not should_skip_dir(os.path.join(dirpath, d))]
                        for dirname in dirnames:
                            repo_path = os.path.join(dirpath, dirname)
                            # repo_name 直接使用最后一级目录的路径
                            repo_name = dirname
                            future_to_repo[executor.submit(self.process_repo, repo_name, repo_path)] = repo_name

            # 等待所有任务完成并收集结果
            for future in tqdm(as_completed(future_to_repo), total=len(future_to_repo)):
                repo_name = future_to_repo[future]
                repo_files[repo_name] = future.result()

        return repo_files

    def process_repo(self, repo_name, repo_path):
        repo_files = []
        sorted_files = self.collect_files(repo_path)
        for filename, file_path in sorted_files:
            content, token_num = self.get_content(file_path)
            if self.bigfile_filter:
                if token_num > 100000:
                    continue
            repo_files.append({
                "file_name": filename,
                "content": content,
                "token_num": token_num
            })
        return repo_files

    def get_content(self, file_path: str):
        after_encode = ""
        line_num = 0
        with open(file_path, 'rb') as f:  # gbk gb2312 gb18030
            for line in f.readlines():
                line_num += 1
                try:
                    line = line.decode("GB18030")
                except UnicodeDecodeError:
                    det_encoder = chardet.detect(line)['encoding']
                    if not det_encoder:
                        print(f"Error: {file_path} have a line no suggest decode !!!")
                        self.unexpected_encoder_num += 1
                        continue
                    try:
                        line = line.decode(det_encoder)
                    except UnicodeDecodeError:
                        self.unexpected_encoder_num += 1
                        continue
                after_encode += line.rstrip() + '\n'
        after_encode = remove_extra_comment(after_encode)
        # after_encode = simplify_comment(after_encode)
        token_ids = self.tokenizer.encode(after_encode)
        token_num = len(token_ids)
        return after_encode, token_num


class CmwDatasetFim:

    def __init__(self, dataset_stage1, fim_rate, fim_spm_rate, seed,
                 tokenizer_path, seq_length, cmw_dataset_output_path):

        self.dataset_stage1 = dataset_stage1
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seq_length = int(seq_length)
        self.seed = seed
        self.tokenizer_path = tokenizer_path

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.eos_token = self.tokenizer.eos_token
        self.prefix_token = "<fim_prefix>"
        self.suffix_token = "<fim_suffix>"
        self.middle_token = "<fim_middle>"
        self.repo_token = "<repo_name>"
        self.file_token = "<file_sep>"
        self.max_span_len = 256  # fim_middle的最大长度
        self.min_span_len = 3  # fim_middle的最小长度
        self.min_file_len = 200  # 进行FIM处理的最小文件长度
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
        raw_dataset = self.dataset_stage1
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

    # pure SPM
    new_sample = prefix_tok + suffix_tok + suffix + middle_tok + prefix + middle + eos_tok

    # if np_rng.binomial(1, fim_spm_rate):
    #     # SPM (variant 2 from FIM paper)
    #     new_sample = prefix_tok + suffix_tok + suffix + middle_tok + prefix + middle + eos_tok
    #
    # else:
    #     # PSM
    #     new_sample = prefix_tok + prefix + suffix_tok + suffix + middle_tok + middle + eos_tok

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


args = arg_parser()


def run_stage1():
    stage1 = CmwDataset(args.dir_name, args.tokenizer_path, args.bigfile_filter, args.project_language)
    # stage1 dataset in Nested Dict format
    stage1_dataset_ND = stage1.traverse_and_collect()
    print("stage1 dataset has been generated.")
    return stage1_dataset_ND


def run_dedup(stage1_dataset_ND):
    stage1_dataset_NJ = convert_nested_dict_to_normjson(stage1_dataset_ND)

    with open("stage1_dataset_nj.json", 'w', encoding='utf-8') as f:
        json.dump(stage1_dataset_NJ, f, indent=4, ensure_ascii=False)

    print("dedup stage start...")

    script_pth = "./run_dedup.sh"
    run_shell_script(script_pth)

    with open("dataset_stage1_dedup.jsonl", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        tmp_js_datas = [json.loads(line) for line in lines]

    with open("stage1_dedup_json", 'w', encoding='utf-8') as f:
        json.dump(tmp_js_datas, f, indent=4, ensure_ascii=False)

    # stage1 dedup nested dict
    stage1_dedup_ND = convert_norm_json_to_nested_dict("stage1_dedup_json")
    return stage1_dedup_ND


def run_stage2(stage1_dedup_ND):
    stage2 = CmwDatasetFim(stage1_dedup_ND, args.fim_rate, args.fim_spm_rate, args.seed,
                           args.tokenizer_path, args.seq_length, args.cmw_dataset_output_path)
    stage2.build_fim_dataset()
    print("FIM dataset has been successfully built")


def run_pipeline():
    start_time = time.time()

    stage1_dataset_ND = run_stage1()
    stage1_dedup_ND = run_dedup(stage1_dataset_ND)
    run_stage2(stage1_dedup_ND)

    print(f"Total time: {time.time() - start_time}")


if __name__ == '__main__':
    run_pipeline()
