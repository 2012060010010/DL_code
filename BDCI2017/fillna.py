import pandas as pd
test_df_final = pd.read_csv("./chandi/submission-Test-mergesevenfinal.csv", index_col=None, header=None)
test_df_final = test_df_final.fillna(";")
# test_df_final = test_df_final.astype(str)
test_df_final.columns = ["row_id", "content", "theme", "sentiment_word", "sentiment_anls"]

test_df11 = pd.read_csv("./chandi/submission-Test-MergeSevenExceptTwo.csv", index_col=None, header=None)
test_df11 = test_df11.astype(str)
test_df11.columns = ["row_id", "content", "theme", "sentiment_word", "sentiment_anls"]


test_df_final.loc[(test_df11.sentiment_word != "nan") & (test_df_final.sentiment_word == ";"), "theme"] = test_df11[(test_df11.sentiment_word != "nan") & (test_df_final.sentiment_word == ";")].loc[:,"theme"]

test_df_final.loc[(test_df11.sentiment_word != "nan") & (test_df_final.sentiment_word == ";"), "sentiment_word"] = test_df11[(test_df11.sentiment_word != "nan") & (test_df_final.sentiment_word == ";")].loc[:,"sentiment_word"]


test_df_final.loc[(test_df11.sentiment_word != "nan") & (test_df_final.sentiment_word == ";"), "sentiment_anls"] = test_df11[(test_df11.sentiment_word != "nan") & (test_df_final.sentiment_word == ";")].loc[:,"sentiment_anls"]

test_df_final.to_csv("./chandi/submission-Test-mergeseven{}.csv".format("fillna"), encoding="utf-8", index=False, columns=["row_id", "content", "theme", "sentiment_word", "sentiment_anls"], header=None)