
1. ***********first step(analyse_data.py)************************
    change_letter()  # feature4

2. ***********second step(analyse_data.py) ***********************
    analyse_post(filename1,filename2)   -----------------(1)
    trans_vector(filename2,filename3) -------------------(2)

    concat_vector('../task3_medium/all_user_vector.txt')

    (1)(2) functions need to be run eight times.
     first time: filename1: '2_Post.txt'   filename2: 'post_raw_group_data.csv'   filename3: 'post_vector.txt'

3. ***********third step(analyse_data.py)*************************
    user_follow_vector()

4. ***********fourth step(analyse_data.py)*************************
    add feature: fans' max fans number
    extract_fans_second_fans('../SMP2017/SMPCUP2017数据集/8_Follow.txt','../task3_medium/follow_vector.txt')

5. ************fifth step(analyse_data.py)*************************
    feature3
    write_feature3_to_file()

6. ************sixth step(analyse_data.py)*****************************************
    #feature4
    stopwords = load_stopwords('../data/stop_words_new.txt')
    all_user = load_alluser('../task3_feature_first/all_user_vector.txt')
    document_dict = make_document_dict('../task3_code/seg_word.txt')  # seg_words.txt is from the task3_code/feature_4_seg.py(running on server)
    extract_all(stopwords,all_user,document_dict)
7. *************seventh step************************
    use LINE(https://github.com/tangjianpku/LINE) to get graph embedding based on '8_Follow.txt'
    command:
        windows: line.exe -train 8_Follow.txt -output embedding_file_128 -binary 0 -size 128 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20
        linux:   ./line -train 8_Follow.txt -output embedding_file_128 -binary 0 -size 128 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20

8. *************eigth step(LR.py)******************************
    write_all_feature_to_file()

9. *************ninth step(analyse_data.py)***********************************************
    extract_ratio()

10. **************tenth step(analyse_data.py)************************************************
    trend_information()

11. *************eleventh step(LR.py)  running on the server*******************************************************
    all_user_dict = load('../task3_feature_final/all_user_dict(395).txt')
    stacking(all_user_dict)




--------------------about features-----------------------------------------------------------------------------------
*********84****************
0 - 11 :   post(12 months)

12 - 23 :  browse(12 months)

24 - 35 :  comment(12 months)

36 - 47 :  vote up(12 months)

48 - 59 :  vote down(12 months)

60 - 71 : favorite(12 months)

72 - 83 : letter(12 months)
**********3**************
84 : fans number

85 : follow number

86 : fans' max fans number(second order relation: used to represent the user importance)
***********16*************
(87 - 102 about the blog the user posts)
87 - 89 : sum browse number , max browse number , mean browse number

90 - 92 : sum comment number , max comment number , mean comment number

93 - 95 : sum vote up number , max vote up number , mean vote up number

96 - 98 : sum vote down number , max vote down number , mean vote down number

99 : mean vote up and vote dowm number

100 - 102 : sum favorite number , max favorite number , mean favorite number
**********128*************
103 - 230 : line(graph embedding)
**********12**************
231 - 233 ： user posted blogs' max length , min length , mean length

234 - 236 : user fans posted blogs' max , min, mean length

237 - 239 : user following people posted blogs' max, min , mean length

240 - 242 : users' lettered people's posted blogs' max , min , mean length
***********************
(243 - 291  7*7 : sum , max , min ,mean ,media, var range)
243 - 249 :  post

250 - 256 :  browse

257 - 263 : comment

264 - 270 : vote up

271 - 277 : vote down

278 - 284 : favorite

285 - 291 : letter
********26***************
292 - 303 : user vote up number / user vote dowm number (12 months)

304 - 315 : user browse number / user comment number (12 months)

316 : lettered number / fans number

317 : letter number / follow number
********77***************
318 - 394  7*11  trendency information

*********1*************************
395 : fans mean fans number



