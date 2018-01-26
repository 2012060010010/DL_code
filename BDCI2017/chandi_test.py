import pandas as pd
from row_apply import senti_gen_with_fasttext
# from method_v4_for_code import pair_postpreprocess
from row_apply import pair_postpreprocess
### generate sentiment value for chandi, test dataframe

chandi_test_df = pd.read_csv("./chandi/submission-Test-mergesevenfillna.csv", header=None)

chandi_test_df.columns = ['row_id', 'content', 'theme', 'sentiment_word', "sentiment_anls"]
chandi_test_df = chandi_test_df.loc[:, ['row_id', 'content', 'theme', 'sentiment_word']]
chandi_test_df = chandi_test_df.fillna(";")
chandi_test_df = chandi_test_df.astype(str)
print('senti_gen_with_fasttext!!')
chandi_test_df = chandi_test_df.apply(senti_gen_with_fasttext, axis=1)
print('pair_postpreprocess!!')
chandi_test_df = chandi_test_df.apply(pair_postpreprocess, axis=1)

chandi_test_df.to_csv("./chandi/submission-Test-mergeseven{}.csv".format("final"), encoding="utf-8-sig", index=False, columns=["row_id", "content", "theme", "sentiment_word", "sentiment_anls"], header=None)

