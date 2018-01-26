#coding:utf8
import pickle
import os
import time
dirpath="../../SMP2017_data/SMPCUP2017_2/cs_word/"
outpath="../../SMP2017_data/SMPCUP2017_2/cs_word_out/"
def cs_word_expand(dirpath,outpath):
    word_merge = []
    cs_word_dict={}
    for root, dirs, files in os.walk(dirpath):
        for f in files:
            print f
            label=f.split('.')[0]
            file = open(root + f, 'r')
            lines = file.readlines()
            file.close()
            word_list = []
            for line in lines:
                wo=line.strip().replace(' ','')
                if len(wo)>1:
                    word_list.append(wo)
            for i in range(len(word_list)):
                w = word_list[i]
                if len(w)>4:
                    w_up = w.upper()
                    w_lo = w.lower()
                    word_list.append(w_up)
                    word_list.append(w_lo)
            word_list = list(set(word_list))
            cs_word_dict[label]=word_list
            print len(word_list)
            with open(outpath + f, 'w') as f:
                f.write('\n'.join(word_list))
            word_merge.extend(word_list)
        word_merge = set(word_merge)
        print len(word_merge)
    with open(outpath + 'cs_word_34.txt', 'w') as f:
        f.write('\n'.join(word_merge))
    with open(outpath + 'label2cs_word_34.pkl', 'w') as ff:
        pickle.dump(cs_word_dict,ff)

cs_word_expand(dirpath, outpath)
time.sleep(0.05)
def cs_word_nodup():
    with open(outpath + 'label2cs_word_34.pkl', 'r') as ff:
        cs_word_dict = pickle.load(ff)

    labels = cs_word_dict.keys()
    label2word = {}

    print len(labels)
    print ' '.join(labels)

    for lab in labels:
        main_words = cs_word_dict[lab]
        print 'label:' + lab
        word_times = {}
        for w in main_words:
            hit_num = 0
            for i in range(0, len(labels)):
                if w in cs_word_dict[labels[i]]:
                    hit_num += 1
            word_times[w] = hit_num
        for m in main_words:
            if word_times[m] > 2:
                print m
                main_words.remove(m)
        label2word[lab] = main_words
    with open(outpath + 'label2cs_word_nodup_34.pkl', 'w') as ff:
        pickle.dump(label2word, ff)
cs_word_nodup()

