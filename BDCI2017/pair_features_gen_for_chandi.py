# with open("./candidatePairList.json", "r") as f:
    # candidate_pair = json.load(f)

# sent = candidate_pair[0][0][0][0]
# pairs = [x[1] for x in candidate_pair[0][0]]
import copy
def pair_features(sent, pairs):
    null = {
            "multi_senti": 0,
            "multi_theme": 0,
            # 主题词共享池特征
            "num_longer_theme": 0,
            "num_shorter_theme": 0,
            "is_other_themes_substring": 0,
            "is_other_themes_extstring": 0,
            "further_than_other_themes": 0,
            "shorter_than_other_themes": 0,
            # 情感词共享池特征
            "num_longer_senti": 0,
            "num_shorter_senti": 0,
            "is_other_sentis_substring": 0,
            "is_other_sentis_extstring": 0,
            "further_than_other_sentis": 0,
            "shorter_than_other_sentis": 0,
        }
    if len(pairs) == 1:
        # 没有答案， 一个空候选对
        features["null"] = null
        return features
    sentis = list(set([_[1] for _ in pairs[:-1]]))
    themes = list(set([_[0] for _ in pairs[:-1]]))
    multi_senti = 1 if len(sentis) > 1 else 0
    multi_theme = 1 if len(themes) > 1 else 0
    themes_features = []
    sentis_features = []
    if multi_theme:
        themes_features = [{} for i in range(len(themes))]
        themes_length = []
        for theme in themes:
            _ = len(theme) if theme != "NULL" else 0
            themes_length.append(_)
        for i, _ in enumerate(themes_length):
            tmp_length = copy.deepcopy(themes_length)
            tmp_length.remove(_)
            num_longer_theme = 0
            num_shorter_theme = 0
            for _other in tmp_length:
                if _ > _other:
                    num_longer_theme += 1
                elif _other > _:
                    num_shorter_theme += 1
            themes_features[i]["num_longer_theme"] = num_longer_theme
            themes_features[i]["num_shorter_theme"] = num_shorter_theme
        
        for i, _ in enumerate(themes):
            tmp_themes = copy.deepcopy(themes)
            tmp_themes.remove(_)
            is_other_themes_substring = 0
            is_other_themes_extstring = 0
            for _other in tmp_themes:
                if _ in _other:
                    is_other_themes_substring = 1
                if _other in _:
                    is_other_themes_extstring = 1
            themes_features[i]["is_other_themes_substring"] = is_other_themes_substring
            themes_features[i]["is_other_themes_extstring"] = is_other_themes_extstring
        
    else:
        # 只有一个NULL 主题词
        themes_features.append(
            {
            # 主题词共享池特征
            "num_longer_theme": 0,
            "num_shorter_theme": 0,
            "is_other_themes_substring": 0,
            "is_other_themes_extstring":0,
            "further_than_other_themes": 0,
            "shorter_than_other_themes": 0,
            })
    if multi_senti:
        sentis_features = [{} for i in range(len(sentis))]
        sentis_length = []
        for senti in sentis:
            _ = len(senti) if senti != "NULL" else 0
            sentis_length.append(_)
        for i, _ in enumerate(sentis_length):
            tmp_length = copy.deepcopy(sentis_length)
            tmp_length.remove(_)
            num_longer_senti = 0
            num_shorter_senti = 0
            for _other in tmp_length:
                if _ > _other:
                    num_longer_senti += 1
                elif _other > _:
                    num_shorter_senti += 1
            sentis_features[i]["num_longer_senti"] = num_longer_senti
            sentis_features[i]["num_shorter_senti"] = num_shorter_senti
        
        for i, _ in enumerate(sentis):
            tmp_sentis = copy.deepcopy(sentis)
            tmp_sentis.remove(_)
            is_other_sentis_substring = 0
            is_other_sentis_extstring = 0
            for _other in tmp_sentis:
                if _ in _other:
                    is_other_sentis_substring = 1
                if _other in _:
                    is_other_sentis_extstring = 1
            sentis_features[i]["is_other_sentis_substring"] = is_other_sentis_substring
            sentis_features[i]["is_other_sentis_extstring"] = is_other_sentis_extstring
    else:
        # 只有一个情感词
        sentis_features.append(
            {
            # 情感词共享池特征
            "num_longer_senti": 0,
            "num_shorter_senti": 0,
            "is_other_sentis_substring": 0,
            "is_other_sentis_extstring":0,
            "further_than_other_sentis": 0,
            "shorter_than_other_sentis": 0,
            })
    if multi_theme:
        senti_maxdistance = {}
        tmp_distance = []
        for s_ in sentis:
            fix = sent.index(s_)
            for t__ in themes:
                if t__ == "NULL":
                    tmp_distance.append(0)
                else:
                    tmp_distance.append(abs(sent.index(t__)-fix))
            max_length = max(tmp_distance)
            senti_maxdistance[s_] = max_length  # 多个主题，针对一个情感词距离他最远的主题词的距离
        
    if multi_senti:
        theme_maxdistance = {}
        tmp_distance = []
        for t_ in themes:
            if t_ == "NULL":
                theme_maxdistance[t_] = 100
                continue
            fix = sent.index(t_)
            for s_ in sentis:
                tl = abs(fix - sent.index(s_))
                tmp_distance.append(tl)
            max_length = max(tmp_distance)
            theme_maxdistance[t_] = max_length  # 多个情感，针对一个主题词距离他最远的情感词的距离

    features = {}
    for i, t_ in enumerate(themes):
        for j, s_ in enumerate(sentis):
            features[t_+" "+s_] = {}
            if t_ != "NULL":
                distance = abs(sent.index(t_)-sent.index(s_))
            else:
                distance = 0
            if multi_theme:
                features[t_+" "+s_]["further_than_other_themes"] = 1 if distance == senti_maxdistance[s_] else 0
                features[t_+" "+s_]["shorter_than_other_themes"] = 1 if distance < senti_maxdistance[s_] else 0
            else:
                features[t_+" "+s_]["further_than_other_themes"] = 0
                features[t_+" "+s_]["shorter_than_other_themes"] = 0
            if multi_senti:
                features[t_+" "+s_]["further_than_other_sentis"] = 1 if distance == theme_maxdistance[t_] else 0
                features[t_+" "+s_]["shorter_than_other_sentis"] = 1 if distance < theme_maxdistance[t_] else 0
            else:
                features[t_+" "+s_]["further_than_other_sentis"] = 0
                features[t_+" "+s_]["shorter_than_other_sentis"] = 0
            features[t_+" "+s_].update(themes_features[i])
            features[t_+" "+s_].update(sentis_features[j])
            features[t_+" "+s_]["multi_senti"] = multi_senti
            features[t_+" "+s_]["multi_theme"] = multi_theme
            assert len(features[t_+" "+s_]) == 14
    features["null"] = null
    assert len(features) == len(pairs)
    return features