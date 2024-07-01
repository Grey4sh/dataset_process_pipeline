#!/usr/bin/env python
# coding=utf-8

"""
data: 5-21-2024 13:29
author: y31506
"""

import argparse
import chardet
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import time
import re
import json

# 要跳过的文件夹后缀列表
skip_suffixes = ["/grpcplugin", "/resource", "/webplugin", "/verifysuite"]
# 所有可能的 trunk 名称
trunk_names = ['v9', 'b75', 'b70', 'b64']


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_name", help="待处理项目路径", default=None)
    parser.add_argument("--stage1_output_path", help="输出一阶段json文件", default=None)
    parser.add_argument("--tokenizer_path", help="tokenizer文件所在目录", default=None)
    parser.add_argument("--bigfile_filter", help="是否对大文件进行剔除", default=False)
    return parser.parse_args()


# 去除函数头，文件头以及修改信息注释
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


def get_trunk_name(path):
    """从路径中精确提取trunk名称"""
    for name in trunk_names:
        if os.path.basename(os.path.normpath(path)) == name:
            return name
    return "unknown"


def should_skip_dir(dir_path):
    """检查目录路径是否以任何一个指定的后缀结尾，如果是则返回 True"""
    for suffix in skip_suffixes:
        if dir_path.endswith(suffix):
            return True
    return False


def simplify_comment(code_string: str):
    comment_pattern = re.compile(r'/\*{1,2}[\s\S]*?\*/', re.DOTALL)
    field_patterns = {
        'Func Name': r'Func Name:\s*(.*?)(?=\n\s*[A-Z]|$)',
        'Description': r'Description:\s*((?:.|\n)*?)(?=\n\s*Input:)',
        'Input': r'Input:\s*((?:.|\n)*?)(?=\n\s*Output:)',
        'Output': r'Output:\s*((?:.|\n)*?)(?=\n\s*Return:)',
        'Return': r'Return:\s*(.*?)(?=\n\s*Caution|$)',
    }

    def process_comment_block(comment_block):
        if comment_block.count('\n') <= 5:
            return comment_block
        if "Func Name:" in comment_block:
            new_comment = "/*\n"
            for key, pattern in field_patterns.items():
                match = re.search(pattern, comment_block, re.DOTALL)
                if match and match.group(1).strip():
                    # Remove excess spaces from the matched content
                    content = ' '.join(match.group(1).strip().splitlines())
                    content = re.sub(r'\s+', ' ', content)  # Replace multiple spaces with a single space
                else:
                    content = ''
                new_comment += f"{key}: {content}\n"
            new_comment += "*/\n"
            return new_comment
        return ""

    new_code_string = re.sub(comment_pattern, lambda match: process_comment_block(match.group(0)), code_string)
    return new_code_string


class CmwDataset:
    def __init__(self, dir_name, raw_dataset_path, tokenizer_path, bigfile_filter):
        self.dir_name = dir_name
        self.raw_dataset_path = raw_dataset_path
        self.tokenizer_path = tokenizer_path
        self.bigfile_filter = bigfile_filter
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.unexpected_encoder_num = 0

    @staticmethod
    def collect_files(repo_path):
        """收集repo中的.c和.h文件，并按要求排序"""
        include_files = []
        lib_files = []
        other_files = []

        for dirpath, dirnames, filenames in os.walk(repo_path):
            # 过滤出需要跳过的目录
            dirnames[:] = [d for d in dirnames if not should_skip_dir(os.path.join(dirpath, d))]

            for filename in filenames:
                if filename.endswith('.c') or filename.endswith('.h'):
                    file_path = os.path.join(dirpath, filename)

                    if '/include/' in dirpath or dirpath.endswith('/include'):
                        include_files.append((filename, file_path))
                    elif '/lib/' in dirpath or dirpath.endswith('/lib'):
                        lib_files.append((filename, file_path))
                    else:
                        other_files.append((filename, file_path))

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

        # 排序各个部分的文件
        sorted_include_files = sort_files(include_files)
        sorted_lib_files = sort_files(lib_files)
        sorted_other_files = sort_files(other_files)

        # 合并所有文件列表
        sorted_files = sorted_include_files + sorted_lib_files + sorted_other_files

        return sorted_files

    def traverse_and_collect(self):
        root = self.dir_name
        trunk_name = get_trunk_name(root)
        if trunk_name == "unknown":
            raise Exception("待处理项目路径中不包含trunk目录")

        repo_files = {}

        for component in tqdm(os.listdir(root)):
            component_path = os.path.join(root, component)
            if os.path.isdir(component_path):
                # 遍历 /src/sbin 和 /src/kernel 目录
                for folder in ["sbin", "kernel"]:
                    target_dir = os.path.join(component_path, f"src/{folder}")
                    if os.path.exists(target_dir):
                        for dirpath, dirnames, filenames in os.walk(target_dir):
                            # 过滤出需要跳过的目录
                            dirnames[:] = [d for d in dirnames if not should_skip_dir(os.path.join(dirpath, d))]
                            # 计算repo名称
                            for dirname in tqdm(list(dirnames)):  # 使用list复制，防止修改dirnames
                                repo_name = f"{trunk_name}-{component}-{folder}-{dirname}"
                                repo_path = os.path.join(dirpath, dirname)
                                if repo_name not in repo_files:
                                    repo_files[repo_name] = []
                                # 收集并排序repo中的文件
                                sorted_files = self.collect_files(repo_path)
                                for filename, file_path in sorted_files:
                                    content, token_num = self.get_content(file_path)
                                    if self.bigfile_filter:
                                        if token_num > 100000:
                                            continue
                                    repo_files[repo_name].append({
                                        "file_name": filename,
                                        "content": content,
                                        "token_num": token_num
                                    })
                            # 清空dirnames以避免进入更深层次的子目录
                            dirnames[:] = []

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

    def save_to_json(self, data):
        with open(self.raw_dataset_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    start_time = time.time()
    args = arg_parser()
    dataset = CmwDataset(args.dir_name, args.stage1_output_path, args.tokenizer_path,
                         args.bigfile_filter)

    collected_files = dataset.traverse_and_collect()
    dataset.save_to_json(collected_files)

    print(f"Total time: {time.time() - start_time}")
