# -*- coding = utf-8 -*-
# @File : utils.py
# @Software : PyCharm
import datetime
import logging
import os
import random
import re

import numpy as np
import torch


def epoch_log(info):
    logging.info("Start A New Epoch" + "\n" + "==========" * 8)
    logging.info(str(info) + "\n")


def init_log():
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists("logs/"):
        os.makedirs("logs/")

    # 创建一个日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 创建文件处理程序，将日志写入文件
    file_handler = logging.FileHandler(f'logs/{nowtime}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # 创建控制台处理程序，将日志打印到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # 将处理程序添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def extract_loss_from_filename(filename):
    """ 从文件名中提取损失值 """
    match = re.search(r'loss_([0-9.]+)\.pt', filename)
    return float(match.group(1)) if match else None


def keep_top_files(folder_path, top_x):
    """ 保留前x个文件，删除其余文件 """
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    # 提取文件名中的损失值，并与文件名一起存储
    files_with_loss = [(file, extract_loss_from_filename(file)) for file in files]
    # 按损失值降序排列文件
    files_sorted = sorted(files_with_loss, key=lambda x: x[1], reverse=True)

    # 保留前x个文件，删除其余文件
    for file, _ in files_sorted[top_x:]:
        os.remove(os.path.join(folder_path, file))
        logging.warning(f"Delete File: {file}")


def get_smallest_loss_model_path(folder_path):
    """ 获取最小损失模型的文件绝对路径 """
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    # 提取文件名中的损失值，并与文件名一起存储
    files_with_loss = [(file, extract_loss_from_filename(file)) for file in files]
    # 移除没有正确损失值的文件（如果有的话）
    files_with_loss = [file for file in files_with_loss if file[1] is not None]
    # 按损失值升序排列文件
    files_sorted = sorted(files_with_loss, key=lambda x: x[1])

    # 获取最小损失模型的文件名
    smallest_loss_file = files_sorted[0][0] if files_sorted else None
    # 返回最小损失模型的绝对路径
    return os.path.join(folder_path, smallest_loss_file) if smallest_loss_file else None
