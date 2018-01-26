import json
import copy
with open("./data/pattern_result_test.json", "r") as f:
    pattern_result = json.load(f)
def rule_extract(row):
    global pattern_result
    # sub_sents = copy.deepcopy(row["sub_sents_tokenized"])
    sentis = row["sentiment_word"][:-1].split(";")
    themes = row["theme"][:-1].split(";")
    values = row["sentiment_anls"][:-1].split(";")
    pairs = [senti+" "+theme + " "+ value for senti, theme, value in zip(sentis, themes, values)]
    for sent, theme, senti, value in pattern_result:
        if sent in row["content"]:
            if senti in sentis:
                for row_senti, row_theme, row_value in zip(sentis, themes, values):
                    if row_senti == senti:
                        pairs.remove(row_senti + " " + row_theme + " " + row_value)
                pairs.append(senti + " " + theme + " " + value)
            else:
                pairs.append(senti + " " + theme + " " + value)
    new_sentis = ";".join([x.split(" ")[0] for x in pairs]) + ";"
    new_themes = ";".join([x.split(" ")[1] for x in pairs]) + ";"
    new_values = ";".join([x.split(" ")[2] for x in pairs]) + ";"
    row["sentiment_word"] = new_sentis
    row["sentiment_anls"] = new_values
    row["theme"] = new_themes
    if sentis != new_sentis:
        print(sentis, themes, values)
        print(row)
    return row