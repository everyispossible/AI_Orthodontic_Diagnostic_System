# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: zhanghongyuan2017@email.szu.edu.cn

import os
import json
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image


from utils.landmarks_utils import check_and_make_dir


class create_index_json:
    def __init__(self,input_dir:str,output_dir:str):
        self.DATASET_PATH=input_dir
        self.OUTPUT_PATH=output_dir
        self.TRAIN_PATH = os.path.join(self.DATASET_PATH, "train")
        self.VALID_PATH=os.path.join(self.DATASET_PATH,'valid')
        self.TEST_PATH = os.path.join(self.DATASET_PATH, "test")
        mappings_file = os.path.join(self.DATASET_PATH, "cephalogram_machine_mappings.csv")
        self.mapping_df = pd.read_csv(mappings_file)

    def _get_pixel_size(self, ceph_id):
        """根据患者ID从CSV中查找对应的像素分辨率 """
        # 假设 CSV 中包含 'cephalogram_id'列以及 'pixel_size' 列
        # 根据 Aariz 论文 Table 2，分辨率单位为 mm/pixel [cite: 75]
        row = self.mapping_df[self.mapping_df['cephalogram_id'].str.contains(ceph_id)]
        if not row.empty:
            return float(row['pixel_size'].values[0])


    def read_landmarks_file(self, ceph_id, split_path):
        """读取初级和高级医生标注并取平均值 [cite: 237, 240]"""
        base_path = os.path.join(split_path, "Annotations", "Cephalometric Landmarks")
        senior_file = os.path.join(base_path, "Senior Orthodontists", f"{ceph_id}.json")

        landmarks=[]
        with open(senior_file,'r') as f:
            data=json.load(f)['landmarks']

            for landmark in data:
                value=landmark.get('value')
                landmarks.append([value.get('x'),value.get('y')])

        return landmarks


    def read_cvm_stage(self,ceph_id,split_path):
        cvm_file=os.path.join(split_path,"Annotations", "CVM Stages", f"{ceph_id}.json")
        with open(cvm_file,'r') as f:
            data=json.load(f)
            cvm_value=data['cvm_stage'].get('value')
        return cvm_value


    def process_split(self, split_name, split_path):
        """处理单个子集（train/valid/test）并生成对应 JSON"""
        result_dict = {}
        # 通过 Cephalograms 文件夹确定患者列表 [cite: 235, 422]
        image_dir = os.path.join(split_path, "Cephalograms")
        extensions=set()
        i=0
        for img in os.listdir(image_dir):
            img_path=os.path.join(image_dir,img)
            if os.path.isfile(img_path):
                pid,ext=os.path.splitext(img)
                if ext:
                    extensions.add(ext)
                result_dict[pid] = {
                    "image_path":img_path,
                    "cvm_stage": self.read_cvm_stage(pid, split_path),
                    "landmarks": self.read_landmarks_file(pid, split_path),
                    "pixel_size": self._get_pixel_size(pid)
                }
                i+=1
        output_file = os.path.join(self.OUTPUT_PATH, f"{split_name}_index.json")
        print(f'格式类型有：{extensions}')
        print(f'{split_name}_index.json,共有{i}条数据')
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=4)
        print(f"已生成: {output_file}")


    def run(self):
        if not os.path.exists(self.OUTPUT_PATH):
            os.makedirs(self.OUTPUT_PATH)

        self.process_split('train',self.TRAIN_PATH)
        self.process_split('valid',self.VALID_PATH)
        self.process_split('test',self.TEST_PATH)


if __name__ == "__main__":
    # 使用示例
    processor = create_index_json(input_dir=r'./Aariz', output_dir=r'./processed_data')
    processor.run()

