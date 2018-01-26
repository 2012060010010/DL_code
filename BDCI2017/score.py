def score_v1(train_df, beta=2):
    """按抽的主题算分，那多个NULL怎么算？"""
    tp = 0 # 主题和情感值均正确数量
    fp = 0 # 主题正确，但是情感值错误数量
    fn_m = 0 #  主题错误，多判 more
    fn_l = 0 # 主题错误，漏判 less
    for i in range(train_df.shape[0]):
        themes = train_df.iloc[i, 2][:-1].split(";")
        sentis = train_df.iloc[i, 3][:-1].split(";")
        values = train_df.iloc[i, 4][:-1].split(";")
        t_dict = {}
        null_values = []
        for theme, value, senti in zip(themes, values, sentis):
            if theme != "NULL":
                t_dict[theme] = (senti, value)
            else:
                null_values.append((senti, value))
        
        pred_themes = train_df.loc[i, "theme"][:-1].split(";")
        pred_values = train_df.loc[i, "sentiment_anls"][:-1].split(";")
        pred_sentis = train_df.loc[i, "sentiment_word"][:-1].split(";")
#         result_dict = {}
        for theme, value, senti in zip(pred_themes, pred_values, pred_sentis):
            if theme in t_dict:
                if t_dict[theme][1] == value and t_dict[theme][0] == senti:
                    tp += 1
                else:
                    fp +=1
                del t_dict[theme]
                continue
            elif theme == "NULL":
                try:
                    null_values.remove((senti, value))
                    tp += 1
                except ValueError:
                    # NULL主题的情感值错误，怎么算？整个主题都错了，还是情感值错了。。如果有null value就是情感值错了，如果没有就是整个主题都错了(多判)
                    if null_values:
                        fp += 1
                    else:
                        fn_m += 1
            else:
                fn_m += 1
        fn_l += len(t_dict)
        fn_l += len(null_values)
    Precision = tp/(tp+fp+fn_m)
    Recall = tp/(tp+fp+fn_l)
    f_value = ((1+beta**2)*Precision*Recall)/(beta**2*Precision+Recall)
    return tp, fp, fn_l, fn_m, Precision, Recall, f_value


def score_v2(train_df, beta=2):
    """按抽的主题算分，那多个NULL怎么算？"""
    tp = 0 # 主题和情感值均正确数量
    fp = 0 # 主题正确，但是情感值错误数量
    fn_m = 0 #  主题错误，多判 more
    fn_l = 0 # 主题错误，漏判 less
    for i in range(train_df.shape[0]):
        themes = train_df.iloc[i, 2][:-1].split(";")
        sentis = train_df.iloc[i, 3][:-1].split(";")
        values = train_df.iloc[i, 4][:-1].split(";")
        answers = []
        for theme, value, senti in zip(themes, values, sentis):
            answers.append((theme, senti, value))
        
        pred_themes = train_df.loc[i, "theme"][:-1].split(";")
        pred_values = train_df.loc[i, "sentiment_anls"][:-1].split(";")
        pred_sentis = train_df.loc[i, "sentiment_word"][:-1].split(";")
#         result_dict = {}
        for theme, value, senti in zip(pred_themes, pred_values, pred_sentis):
            try:
                answers.remove((theme, senti, value))
                tp += 1
            except ValueError:
                fn_m += 1
        fn_l += len(answers)
    Precision = tp/(tp+fp+fn_m)
    Recall = tp/(tp+fp+fn_l)
    f_value = ((1+beta**2)*Precision*Recall)/(beta**2*Precision+Recall)
    return tp, fp, fn_l, fn_m, Precision, Recall, f_value


def score_v3(train_df, beta=2):
    tp = 0 # 主题和情感值均正确数量
    fp = 0 # 主题正确，但是情感值错误数量
    fn_m = 0 #  主题错误，多判 more
    fn_l = 0 # 主题错误，漏判 less
    for i in range(train_df.shape[0]):
        themes = train_df.iloc[i, 2][:-1].split(";")
        sentis = train_df.iloc[i, 3][:-1].split(";")
        values = train_df.iloc[i, 4][:-1].split(";")
        answers = []
        for theme, value, senti in zip(themes, values, sentis):
            answers.append((theme, senti, value))
        t_dict = {}
        null_values = []
        for theme, value, senti in zip(themes, values, sentis):
            if theme != "NULL":
                t_dict[theme] = (senti, value)
            else:
                null_values.append((senti, value))
        
        pred_themes = train_df.loc[i, "theme"][:-1].split(";")
        pred_values = train_df.loc[i, "sentiment_anls"][:-1].split(";")
        pred_sentis = train_df.loc[i, "sentiment_word"][:-1].split(";")
#         result_dict = {}
        for theme, value, senti in zip(pred_themes, pred_values, pred_sentis):
            try:
                answers.remove((theme, senti, value))
                tp += 1
            except ValueError:
                if theme in t_dict:
                    fp +=1
                    fn_l -= 1
                    continue
                elif theme == "NULL":
                    if null_values:
                        fp += 1
                        fn_l -= 1
                    else:
                        fn_m += 1
                else:
                    fn_m += 1
        fn_l += len(answers)
    Precision = tp/(tp+fp+fn_m)
    Recall = tp/(tp+fp+fn_l)
    f_value = ((1+beta**2)*Precision*Recall)/(beta**2*Precision+Recall)
    return tp, fp, fn_l, fn_m, Precision, Recall, f_value



def score_v3_with_debug(train_df, beta=2):
    tp = 0 # 主题和情感值均正确数量
    fp = 0 # 主题正确，但是情感值错误数量
    fp_samples_w = []
    fp_samples_v = []
    fn_m = 0 #  主题错误，多判 more
    fn_m_samples = []
    fn_l = 0 # 主题错误，漏判 less
    fn_l_samples = []
    for i in range(train_df.shape[0]):
        themes = train_df.iloc[i, 2][:-1].split(";")
        sentis = train_df.iloc[i, 3][:-1].split(";")
        values = train_df.iloc[i, 4][:-1].split(";")
        answers = []
        for theme, value, senti in zip(themes, values, sentis):
            answers.append((theme, senti, value))
        t_dict = {}
        null_dict = {}
        null_values = []
        for theme, value, senti in zip(themes, values, sentis):
            if theme != "NULL":
                t_dict[theme] = (senti, value)
            else:
                null_values.append((senti, value))
                null_dict[senti] = value
        
        pred_themes = train_df.loc[i, "theme"][:-1].split(";")
        pred_values = train_df.loc[i, "sentiment_anls"][:-1].split(";")
        pred_sentis = train_df.loc[i, "sentiment_word"][:-1].split(";")
        if sentis == [""]:
            fn_m += len(pred_themes)
            if pred_themes[0]:
                fn_m_samples.extend([(pt, pv, ps) for pt, pv, ps in zip(pred_themes, pred_values, pred_sentis)])
            continue
        if [""] == pred_sentis:
            fn_l += len(themes)
            if sentis[0]:
                fn_l_samples.extend(answers)
            continue
#         result_dict = {}
        for theme, value, senti in zip(pred_themes, pred_values, pred_sentis):
            try:
                answers.remove((theme, senti, value))
                tp += 1
            except ValueError:
                if theme in t_dict:
                    fp +=1
                    fn_l -= 1
                    if t_dict[theme][0] == senti:
                        fp_samples_v.append((theme, value, senti))
                    else:
                        fp_samples_w.append((theme, value, senti))
                    continue
                elif theme == "NULL":
                    if null_values:
                        fp += 1
                        if senti in null_dict:
                            fp_samples_v.append((theme, value, senti))
                        else:
                            fp_samples_w.append((theme, value, senti))
                        fn_l -= 1
                    else:
                        fn_m += 1
                        assert "" != senti
                        fn_m_samples.append((theme, value, senti))
                else:
                    fn_m += 1
#                     print(pred_themes, pred_values, pred_sentis)
                    assert "" != senti
                    fn_m_samples.append((theme, value, senti))
        fn_l += len(answers)
        fn_l_samples.extend(answers)
    Precision = tp/(tp+fp+fn_m)
    Recall = tp/(tp+fp+fn_l)
    Precision_chandi = (tp+fp-len(fp_samples_w))/(tp+fp-len(fp_samples_w)/2.0+fn_m)
    Recall_chandi = (tp+fp-len(fp_samples_w))/(tp+fp-len(fp_samples_w)/2.0+fn_l)
    f_value = ((1+beta**2)*Precision*Recall)/(beta**2*Precision+Recall)
    f_value_chandi = ((1+beta**2)*Precision_chandi*Recall_chandi)/(beta**2*Precision_chandi+Recall_chandi)
    return tp, fp, fn_l, fn_m, Precision, Recall, f_value, f_value_chandi, fp_samples_w, fp_samples_v, fn_m_samples, fn_l_samples

# 单纯改变