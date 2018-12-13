#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-12-6 下午3:00

import nltk
from data_helper import load_qa_data


def lemmatize():
    wn_lemmatizer = nltk.stem.WordNetLemmatizer()
    data_sets = ['traindata', 'validationdata', 'eval1_unlabelled']
    for set_name in data_sets:
        fin_path = 'data/raw/{}.tsv'.format(set_name)
        fout_path = 'data/lemmatized/{}.tsv'.format(set_name)
        with open(fin_path, 'r', encoding='utf8', errors='ignore') as fin, open(fout_path, 'w', encoding='utf8', errors='ignore') as fout:
            fin.readline()
            if set_name=='eval1_unlabelled':
                i = 0
            else:
                i = 1
            for line in fin:
                line_info = line.strip().split('\t')
                q_id = line_info[i]
                question = line_info[i + 1]
                a_id = line_info[i + 4 + (i - 1)]
                answer = line_info[i + 2]
                question = ' '.join(map(lambda x: wn_lemmatizer.lemmatize(x), nltk.word_tokenize(question)))
                answer = ' '.join(map(lambda x: wn_lemmatizer.lemmatize(x), nltk.word_tokenize(answer)))
                if set_name != 'eval1_unlabelled':
                    label = line_info[i + 3]
                    # print(label)
                    fout.write('\t'.join([q_id, question, a_id, answer, label]) + '\n')
                else:
                    fout.write('\t'.join([q_id, question, a_id, answer]) + '\n')
        print("Finished {}".format(set_name))


def gen_train_triplets(same_q_sample_group):
    question = same_q_sample_group[0].question
    pos_answers = [sample.answer for sample in same_q_sample_group if sample.label == 1]
    neg_answers = [sample.answer for sample in same_q_sample_group if sample.label == 0]
    for pos_answer in pos_answers:
        for neg_answer in neg_answers:
            yield question, pos_answer, neg_answer


def gen_train_samples():
    qa_samples = load_qa_data('data/lemmatized/WikiQA-train.tsv')
    with open('data/lemmatized/WikiQA-train-triplets.tsv', 'w', encoding='utf8', errors='ignore') as fout:
        same_q_samples = []
        for qa_sample in qa_samples:
            if len(same_q_samples) == 0 or qa_sample.q_id == same_q_samples[0].q_id:
                same_q_samples.append(qa_sample)
            else:
                for question, pos_ans, neg_ans in gen_train_triplets(same_q_samples):
                    fout.write('{}\t{}\t{}\n'.format(question, pos_ans, neg_ans))
                same_q_samples = [qa_sample]
        if len(same_q_samples) > 0:
            for question, pos_ans, neg_ans in gen_train_triplets(same_q_samples):
                fout.write('{}\t{}\t{}\n'.format(question, pos_ans, neg_ans))

if __name__ == '__main__':
    lemmatize()
    # gen_train_samples()