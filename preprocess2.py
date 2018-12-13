#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-12-6 下午3:00

import nltk
from data_helper import load_qa_data
import numpy as np


def lemmatize():
    wn_lemmatizer = nltk.stem.WordNetLemmatizer()
    data_sets = ['eval1_unlabelled']
    for set_name in data_sets:
        fin_path = 'data/raw/{}.tsv'.format(set_name)
        fout_path = 'data/lemmatized/{}.tsv'.format(set_name)
        with open(fin_path, 'r', encoding='utf8', errors='ignore') as fin, open(fout_path, 'w', encoding='utf8', errors='ignore') as fout:
            fin.readline()
            if set_name == 'eval1_unlabelled':
                i = 0
            else:
                i = 1
            print(i)
            for line in fin:
                line_info = line.strip().split('\t')
                q_id = line_info[i]
                question = line_info[i + 1]
                a_id = line_info[i + 4 + (i - 1)]
                answer = line_info[i + 2]
                question = ' '.join(map(lambda x: wn_lemmatizer.lemmatize(x), nltk.word_tokenize(question)))
                answer = ' '.join(map(lambda x: wn_lemmatizer.lemmatize(x), nltk.word_tokenize(answer)))
                # label = None
                if set_name != 'eval1_unlabelled':
                    label = line_info[i + 3]
                    # print(label)
                    fout.write('\t'.join([q_id, question, a_id, answer, label]) + '\n')
                else:
                    fout.write('\t'.join([q_id, question, a_id, answer]) + '\n')
                if np.random.random() < 0.00001:
                    print(q_id, question, a_id, answer)
        print("Finished {}".format(set_name))


if __name__ == '__main__':
    lemmatize()
    # gen_train_samples()