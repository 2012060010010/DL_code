HLT-HITSZ-task1-代码说明文档
Author：陈荻-441023012@qq.com

总体代码分为3个部分，分别为
01_task1_Ensemble，
02_task1_TitleSegmentation，
03_task1_TitleSimilarity，
按照数字编号按顺序运行即可。

分词工具包：
1.THULAC (http://thulac.thunlp.org/) -> 用于分词
2.jieba -> 用于计算tf-idf (注意：由于已使用THULAC分词，在使用jieba计算tf-idf过程中，不再进行分词！)

所有代码采用的词表来源包括：
1.搜狗细胞词库
2.SMP CUP 2017任务1训练集的keywords(即tmp目录下的smpLabel.txt)
3.清华大学开放中文词库THUOCL(http://thuocl.thunlp.org/#IT)

01_task1_Ensemble：
-> 首先从博客标题中提取keywords，然后对博客全文分词后的每个keywords按照tf-idf值排序。
-> 若标题keywords在tf-idf-top10之内或者在smpLabel词表之中，可作为最终keywords候选。
-> 筛选后的标题keywords可能多于3个或者少于3个，则相应在博客全文keywords根据tf-idf值排序再一次筛选或者补充。此过程中，若keywords在smpLabel词表中，其排序优先级更高。

02_task1_TitleSegmentation：
-> 将所有的博客文章的标题分词(100w)。

03_task1_TitleSimilarity：
-> 计算测试集中的标题与训练集中的标题相似度，对最终结果进行优化。