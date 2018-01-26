问题二基本思路：
   1、根据用户操作记录（浏览，发布等）抽取用户有关文档，存入一个dict中，{user:doc_ID list} 
   2、通过搜索百度文库的相关专业术语表，生成标签和关键词对应关系。
   3、利用标签词汇在用户文档中匹配，计算各标签在文档标题，英文部分和正文的影响强度（频率），从而确定
每个用户和标签相关行，形成102维向量。
   4、利用文档全集训练文档向量,然后根据用户相关文档及其向量通过平均文档向量对用户进行表示，
最后得到用户和文档向量对应关系。
   5、利用两方面特征对用户进行建模，组合两种向量，利用xgboost工具训练模型，对用户兴趣进行预测。
其中对于多标签处理方式为重复训练样例，从而变成单标签问题。将预测概率前三位的标签作为用户预测标签。

文件执行顺序：
   1、label_word_process.py 
   2、jieba_seg的seg_blog.py (运行时间大概4h)
   3、paragraph2vec.py     （运行时间大概1h）
   4、extract_blog_for_user.py  （运行时间大概0.5h）
   5、word_kaf_filter.py 
   6、user_presentation.py  （占用内存很大，普通机器可能吃力）
   7、xgb_train.py
   8、over_sample_train.py
   9、xgb_split_3.py
   10、test_predict.py  （跳过特征提取过程，直接利用特征进行预测，请直接运行）
  
可能用到的库： (python2.7)
   1、numpy，pandas，heapq
   2、jieba，gensim，sklearn，xgboost等。
