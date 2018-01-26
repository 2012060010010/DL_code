from collections import Counter
import pandas as pd
from row_apply import content_filter
from row_apply import tokenized_sub_sents
from row_apply import content_cut

test_df = pd.read_csv("./chandi/submission-Test-mergesevenfinal.csv", header=None)
test_df = test_df.fillna(";")
test_df = test_df.astype("str")
test_df.columns = ['row_id', 'content', 'theme', 'sentiment_word', "sentiment_anls"]
test_df.iloc[:, 1] = test_df.apply(content_filter, axis=1)
test_df.iloc[:, 1] = test_df.apply(content_cut, axis=1)
test_df = test_df.apply(tokenized_sub_sents, axis=1)


with open("./data/neg_adv_after_manual_filter.txt", "r") as f:
    neg_adv = [x.strip() for x in f.readlines()]
neg_adv = set(neg_adv)


c = 0
for i in range(test_df.shape[0]):
    themes = test_df.loc[i, "theme"]
    themes = themes.replace("NULL;", "")
    themes = themes[:-1].split(";")
    if len(themes) != len(set(themes)):
        themes_counter = Counter(themes)
        rep_themes = [x[0] for x in themes_counter.most_common() if x[1]>1]
        rep_sentis = []
        themes = test_df.loc[i, "theme"]
        themes = themes[:-1].split(";")
        sentis = test_df.loc[i, "sentiment_word"]
        sentis = sentis[:-1].split(";")
        canbe_filter = []
        for theme in rep_themes:
            for t, s in zip(themes, sentis):
                if t == theme:
                    rep_sentis.append(s)
            for sub in test_df.loc[i, "sub_sents_tokenized"]:
                sub = "".join(sub)
#                 print(rep_sentis)
                if theme in sub and rep_sentis[0] in sub and rep_sentis[1] in sub and (rep_sentis[0] in rep_sentis[1] or rep_sentis[1] in rep_sentis[0]):
                    if len(rep_sentis[0]) > len(rep_sentis[1]):
                        tmp = rep_sentis[0].replace(rep_sentis[1], "")
                        shorter = rep_sentis[1]
                    else:
                        tmp = rep_sentis[1].replace(rep_sentis[0], "")
                        shorter = rep_sentis[0]
                    if tmp in neg_adv:   
#                         print(test_df.loc[i, ['row_id', 'content', 'theme', 'sentiment_word', 'sentiment_anls']].values)
                        c += 1
                        canbe_filter.append(theme+ " " +shorter)
        if len(canbe_filter) > 0:
            sentis = test_df.loc[i, "sentiment_word"][:-1].split(";")
            themes = test_df.loc[i, "theme"][:-1].split(";")
            values = test_df.loc[i, "sentiment_anls"][:-1].split(";")
            new_sentis = []
            new_themes = []
            new_values = []
            for t, w, v in zip(themes, sentis, values):
                if t+" " + w not in canbe_filter:
                    new_sentis.append(w)
                    new_themes.append(t)
                    new_values.append(v)
            new_sentis = ";".join(new_sentis) + ";"
            new_themes = ";".join(new_themes) + ";"
            new_values = ";".join(new_values) + ";"
            if len(test_df.loc[i, "theme"]) != len(new_themes):
                print(test_df.loc[i, ['row_id', 'content', 'theme', 'sentiment_word', 'sentiment_anls']].values)
                test_df.loc[i, "sentiment_word"] = new_sentis
                test_df.loc[i, "sentiment_anls"] = new_values
                test_df.loc[i, "theme"] = new_themes
                print("\n")
                print(test_df.loc[i, ['row_id', 'content', 'theme', 'sentiment_word', 'sentiment_anls']].values)
    else:
        continue
		
def pair_postpreprocess_del_cu(row):
    # 在产生答案pair之后，进行一些后处理，比如去重
    themes = row["theme"]
    senti_w = row["sentiment_word"]
    senti_v = row["sentiment_anls"]
    if senti_v == senti_w:
        row["theme"] = ""
        row["sentiment_word"] = ""
        row["sentiment_anls"] = ""
    return row

new_test_df = test_df.apply(pair_postpreprocess_del_cu, axis=1)
new_test_df.to_csv("./chandi/submission-Test-mergeseven{}.csv".format("FINAL"), encoding="utf-8", index=False, columns=["row_id", "content", "theme", "sentiment_word", "sentiment_anls"], header=None)
