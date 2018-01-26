### analysis for chandi valid
import pandas as pd
from collections import Counter
from method_v4_for_code import senti_gen_with_fasttext
from method_v4_for_code import pair_postpreprocess
from method_v4_for_code import score_v3_with_debug

valid_df_chandi = pd.read_csv("./chandi/semiCompetition1201/submission-Valid-BIEonly-addTSN-addSubstringTag-addResultCount-addCharTagProb-addPairFeature.csv")
valid_df_chandi.columns = ['row_id', 'content-评论内容', 'theme-主题', 'sentiment_word-情感关键词','sentiment_anls-情感正负面', 'theme', 'sentiment_word', "sentiment_anls"]
valid_df_chandi = valid_df_chandi.loc[:,['row_id', 'content-评论内容', 'theme-主题', 'sentiment_word-情感关键词', 'sentiment_anls-情感正负面', 'theme', 'sentiment_word',]]
valid_df_chandi = valid_df_chandi.fillna(";")
valid_df_chandi = valid_df_chandi.astype(str)

valid_df_chandi = valid_df_chandi.apply(senti_gen_with_fasttext, axis=1)
valid_df_chandi = valid_df_chandi.apply(pair_postpreprocess, axis=1)

tp, fp, fn_l, fn_m, P, R, F_value, F_value_chandi, fp_samples_chandivalid_w, fp_samples_chandivalid_v, \
                        fn_m_samples_chandivalid, fn_l_samples_chandivalid = score_v3_with_debug(valid_df_chandi, beta=1)

print("Senti True Prediction:{} Senti False Prediction:{} Num Less Theme:{} Num More Theme:{} Precision:{} Recall:{}"
      " F:{} F_chandi:{}".format(tp, fp, fn_l, fn_m, P, R, F_value, F_value_chandi))
print("Senti value Error:{} Senti word Error:{}".format(len(fp_samples_chandivalid_v), len(fp_samples_chandivalid_w)))
fn_m_samples_chandivalid = Counter(fn_m_samples_chandivalid)
fn_l_samples_chandivalid = Counter(fn_l_samples_chandivalid)
fp_samples_chandivalid_w = Counter(fp_samples_chandivalid_w)
fp_samples_chandivalid_v = Counter(fp_samples_chandivalid_v)
