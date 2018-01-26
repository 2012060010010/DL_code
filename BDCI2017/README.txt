运行环境：
ubuntu 
py2.和py3.6

依赖包：
py3
   fasttext
   gensim
   pandas
   sklearn-crfsuite
   tensorflow
   
py2:

sudo pip install sklearn-crfsuite
sudo apt-get install python-tk
sudo pip install jieba
sudo pip install Cython
sudo pip install fasttext
sudo pip install python-Levenshtein
sudo pip install gensim
sudo pip install matplotlib
sudo pip install pyltp
sudo pip install xgboost
sudo pip install pandas
sudo pip install scrapy
sudo pip install flask
sudo pip install numpy
sudo pip install scipy
sudo pip install textrank4zh
sudo pip install scikit-learn
sudo pip install ipython
sudo pip install jupyter notebook
sudo pip install tensorflow


运行流程：
 生成特征：
      py3:
      code/method_v4_for_code.py
	  code/sent_has_ans/eval.py
训练模型：
	  py2:
	  查看 code/chandi/README.txt 即可复现
	  
处理答案：
	 py3：
	 
	  code/fillna.py
	  code/chandi_test.py
	  code/last_process.py
	  
	最终结果文件：
	    code/chandi/submission-Test-mergesevenFINAL.csv
