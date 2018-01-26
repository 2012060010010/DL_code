import pickle
dirpath='../../SMP2017_data/SMPCUP2017'
train_feature_file=dirpath+"_2/feature_doc/usr_word_feature_102_all.pkl"
train_file=dirpath+"_2/SMPCUP2017_TrainingData_Task2.txt"
valid_file=dirpath+"_valid/SMPCUP2017_ValidationSet_Task2.txt"
test_file=dirpath+"_valid/SMPCUP2017_TestSet_Task2.txt"
usr_docvec_file=dirpath+'_2/feature_doc/user2docvec_dbow.pkl'

user_tvt=[]
with open(train_file) as f:
    for line in f :
        user_tvt.append(line.strip().split('\001')[0])
with open(valid_file) as f:
    for line in f :
        user_tvt.append(line.strip())
with open(test_file) as f:
    for line in f :
        user_tvt.append(line.strip())
# with open(dirpath+"_2/user_blog_match/usr_ID_tvt.pkl",'w') as f:
#     pickle.dump(user_tvt,f)
print len(user_tvt)
with open(train_feature_file, "r") as ff:
    user_feature_dict=pickle.load(ff)
with open(usr_docvec_file, "r") as ff:
    user_docvec_dict=pickle.load(ff)

user_feature_tvt={}
user_docvec_tvt={}
for u in user_tvt:
    user_feature_tvt[u]=user_feature_dict[u]
    user_docvec_tvt[u]=user_docvec_dict[u]
print len(user_feature_tvt)
print len(user_docvec_tvt)
# with open(dirpath+"_2/feature_doc/usr_word_feature_102_tvt.pkl",'w') as f:
#     pickle.dump(user_feature_tvt,f)
# with open(dirpath+"_2/feature_doc/user2docvec_dbow_tvt.pkl",'w') as f:
#     pickle.dump(user_docvec_tvt,f)