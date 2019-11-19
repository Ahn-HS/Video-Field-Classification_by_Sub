import time, re, operator, os, openpyxl

import pandas as pd
import numpy as np

# Count, TF-IDF 패키지
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# 빈도수 확인 패키지
import collections
from collections import defaultdict
from collections import Counter

# 학습용, 평가용 데이터 분할 cross validation
from sklearn.model_selection import train_test_split

# 한국어 형태소 분석 및 예제 패키지
from konlpy.tag import Twitter
from konlpy.corpus import kolaw

# 영어 형태소 분석 패키지
import nltk
from nltk.collocations import *
from nltk.tokenize import word_tokenize

# gensim 에서 word2vec 사용
from gensim.models import word2vec
import gensim.models as g

# 한국어 word rank 패키지 사용
from krwordrank.word import KRWordRank


############# initialize
pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 500)

dic_key_name = ["한글", "영어", "과학", "자연관찰", "수학", "영어동요", "동화", "사회관계", "예술경험", "건강안전", "동요"]

def keyword_extraction_by_condition():
    data_xls = pd.read_excel('resource/영역_title_keyword_id_190819_notnan_space.xlsx', 'Sheet1', index_col=None)
    cress_exel_file = openpyxl.load_workbook('resource/영역_title_keyword_id_190819_notnan_space.xlsx', data_only=True)
    cress_sheet = cress_exel_file.get_sheet_by_name("Sheet1")

    dic_list = [{} for i in range(11)]
    from collections import ChainMap

    for r_ori_index, row in enumerate(cress_sheet.rows):
        ''' condition에 따라서 데이터 로드 및 저장 '''
        lst_tmp_split = row[2].value.split(" ")
        if r_ori_index == 0 or lst_tmp_split == []:
            continue

        # 리스트 속 딕셔너리 키에 접근 할 수 있는 함수
        chain_map = ChainMap(*dic_list)
        # print(chain_map["키값"])
        # {**dic_list[0], **dic_list[1]}

        for word in lst_tmp_split:
            if word in dic_list[dic_key_name.index(row[0].value)]:
                dic_list[dic_key_name.index(row[0].value)][word] = dic_list[dic_key_name.index(row[0].value)][word] + 1
                # dic[word] = dic[word] + 1
            else:
                dic_list[dic_key_name.index(row[0].value)][word] = 1

    #################################################################################
    for idx, _dic in enumerate(dic_list):
        # sorted_freq_result = sorted(_dic.items(), key=operator.itemgetter(1), reverse=True)

        with open('analysis/' + dic_key_name[idx] + '_frequency.txt', 'w', encoding='utf-8') as file:
            file.writelines(" ".join(list(_dic.keys())))
    #################################################################################

    #################################################################################
    # sorted_freq_result = sorted(_dic.items(), key=operator.itemgetter(1), reverse=True)
    # with open('analysis/keyword_by_condition.txt', 'w', encoding='utf-8') as file:
    #     for idx, sentence in enumerate(string_list):
    #         print(sentence)
    #         if idx == 10:
    #             file.writelines(sentence)
    #################################################################################


def korean_pos_tokenize(raw_data):
    spliter = Twitter()
    result = spliter.pos(raw_data, norm=True, stem=True)
    # result 는 dic 타입으로 출력이 되고 key 값은 잘려진 문자열, value 형태소 종류(예 : 명사, 동사, 관형사, 부사)
    noun_adj_verb = []
    for sentence in result:
        # 명사, 동사 형태소만 추출함
        if sentence[1] in ['Noun', 'Verb']:
            noun_adj_verb.append(sentence[0])

    return noun_adj_verb


def stopword_remove(data_xls):
    corpus_list = []
    d_corpus_list = []
    stop_words = ["하다", "거", "않다", "되다"]
    remove_words = ["VS", "vs", "1화", "2화", "3화", "4화", "5화", "6화", "7화", "8화", "9화", "10화", "1회",
                    "2회", "3회", "4회", "5회", "6회", "7회", "8회", "9회", "10회", "를", "있다", "수", "것"]
    for name in dic_key_name:
        condi = data_xls.loc[data_xls["영역"] == name]
        # condi = train_set.loc[train_set["영역"] == name]
        corlist = []
        # 중복된 단어 없이 추가
        # for word in condi["키워드"].values:
        #     word = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'^0-9]', '', word)
        #     corlist.extend(word.split(' '))
        for word in condi["제목"].values:
            for remove_word in remove_words:
                word = word.replace(remove_word, "")
            word = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'^0-9]', '', word)
            corlist.extend(word.split(' '))
        for word in condi["자막"].values:
            word = str(word)
            if not word.strip() == "nan":
                for remove_word in remove_words:
                    word = word.replace(remove_word, "")
                word = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'^0-9]', '', word)
                tmp = korean_pos_tokenize(word)
                corlist.extend(tmp)
        for check_word in corlist[:]:
            if check_word in stop_words:
                corlist.remove(check_word)
        corpus_list.append(' '.join(set(corlist)))
        d_corpus_list.append(' '.join([x for x in corlist if x]))
    #################################################

    ############### 영어와 숫자 제거 ##################
    # remove_num_doc = re.sub(r'\d', '', file_text)
    # remove_eng_doc = re.sub('[a-zA-Z]', '', remove_num_doc)
    # doc_split = remove_eng_doc.split("\n")
    ##################################################

    return corpus_list, d_corpus_list


def Count_TFIDF_based_featuer_extracion():
    dic_key_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # 첫 문자 '\ufeff' 제거
    file = open('analysis/keyword_by_condition2.txt', 'r', encoding='utf-8-sig', newline='\n')
    file_text = file.read()

    from sklearn.model_selection import StratifiedShuffleSplit
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    data_xls = pd.read_excel('resource/영역_title_keyword_id_190819_notnan_space_sub.xlsx', 'Sheet1', index_col=None)
    train_set, test_set = split_set(data_xls, 0.1)

    ###################### 데이터 빈도수 bar plot ##########################
    font_location = 'C:/Windows/Fonts/H2HDRM.ttf'  # For Windows
    font_name = fm.FontProperties(fname=font_location).get_name()
    matplotlib.rc('font', family=font_name)

    df_test = data_xls["영역"].value_counts()
    plt.figure(figsize=(12, 4))
    sns.barplot(df_test.index, df_test.values, alpha=0.8)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Product', fontsize=12)
    # plt.xticks(rotation=90)
    plt.xticks()
    plt.show()
    ######################

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, test_index in split.split(data_xls, data_xls["영역"]):
        strat_train_set = data_xls.loc[train_index]
        strat_test_set = data_xls.loc[test_index]

    ####### 데이터 확인 #######
    # print(strat_test_set["영역"].value_counts() / len(strat_test_set))
    # print(strat_test_set["영역"].value_counts())
    #
    # print("========== test data set ==========")
    # print(test_set["영역"].value_counts() / len(data_xls))
    # print(test_set["영역"].value_counts())
    # print("===================================")
    ########################

    corpus_list, d_corpus_list = stopword_remove(data_xls)

    ##################### TF-IDF ##############################
    # tf-idf는 DTM 출ㄺ
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vec = tfidf_vectorizer.fit_transform(corpus_list)

    word2id = defaultdict(lambda: 0)
    for idx, feature in enumerate(tfidf_vectorizer.get_feature_names()):
        word2id[feature] = idx

    for i, sentence in enumerate(corpus_list):
        # print('========= document[ %s ] =========' % dic_key_name[i])
        tmp = [(token, tfidf_vec[i, word2id[token]]) for token in sentence.split()]
        sor = sorted(tmp, key=operator.itemgetter(1), reverse=True)
        # sor = sorted(sor, key=operator.itemgetter(0), reverse=False)
        # print(sor)
    ###########################################################

    ##################### COUNT ###############################
    print("count based ")
    count_vectorizer = CountVectorizer()
    count_vec = count_vectorizer.fit_transform(corpus_list)

    for i, sentence in enumerate(d_corpus_list):
        # print('========= document[ %s ] =========' % dic_key_name[i])
        word = sentence.split(" ")
        x_count = collections.Counter(word)
        # print(x.most_common())
        # print()
    ###########################################################

    ############ 각 영역 서로의 유사도 ############
    # tfidf = TfidfVectorizer().fit_transform(corpus_list)
    # print(tfidf.shape)
    # tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_list)

    # document_distances = (tfidf_matrix * tfidf_matrix.T)
    # print(document_distances.get_shape())
    # print(document_distances.toarray())
    ##############################################


def Count_TFIDF_based_classification():
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    # df = pd.read_excel('resource/영역_title_keyword_id_190819.xlsx')
    # train_X, test_X, train_Y, test_Y = train_test_split(df["키워드"], df["영역"], test_size=0.1, random_state=3)

    ###############################################
    # total = 0
    # sgd_model = SGDClassifier(loss='log')
    # sgd_model.fit(tfidf_vec, dic_key_name)
    # X_pred = tfidf_vectorizer.transform(test_set["키워드"])
    # Y_pred = sgd_model.predict(X_pred)
    # total = accuracy_score(test_set["영역"], Y_pred)
    # print("tfidf_logistic regression test set accuracy : ", total)
    #
    # print(classification_report(test_set["영역"], Y_pred, target_names=dic_key_name))
    #
    # conf_mx = confusion_matrix(test_set["영역"], Y_pred)
    # plt.matshow(conf_mx, cmap=plt.cm.gray)
    # plt.show()
    ################################################

    ################################################
    # total = 0
    # svm_model = SGDClassifier(loss='hinge', penalty="l2")
    # svm_model.fit(tfidf_vec, dic_key_name)
    # X_pred = tfidf_vectorizer.transform(test_set["키워드"])
    # Y_pred = svm_model.predict(X_pred)
    # total = accuracy_score(test_set["영역"], Y_pred)
    # print("tfidf_svm test set accuracy : ", total)
    #
    # print(classification_report(test_set["영역"], Y_pred, target_names=dic_key_name))
    ################################################

    import scikitplot as skplt

    ################################################
    # total = 0
    # perceptron_model = SGDClassifier(loss='perceptron')
    # perceptron_model.fit(tfidf_vec, dic_key_name)
    # X_pred = tfidf_vectorizer.transform(test_set["키워드"])
    # Y_pred = perceptron_model.predict(X_pred)
    # total = accuracy_score(test_set["영역"], Y_pred)
    # print("word2vec_dnn test set accuracy : ", total)
    #
    # print(classification_report(test_set["영역"], Y_pred, target_names=dic_key_name))
    #
    # conf_mx = confusion_matrix(test_set["영역"], Y_pred)
    # plt.matshow(conf_mx, cmap=plt.cm.gray)
    # plt.show()
    # # predicted = model_DNN.predict_proba(X_test_tfidf)
    # skplt.metrics.plot_roc_curve(test_set["영역"], Y_pred)
    # plt.show()
    ################################################

    #####################################
    # lr_model = LogisticRegression(solver='sag', multi_class='ovr', max_iter=10)
    # lr_model.fit(tfidf_vec, dic_key_num)
    # X_pred = tfidf_vectorizer.transform(test_X)
    # Y_pred = lr_model.predict(X_pred)
    # total += accuracy_score(test_Y, Y_pred)
    # print("tfidf multi test set accuracy : ", total)
    ####################################

    print()

    #########################################
    # total = 0
    # c_nb_model = MultinomialNB()
    # c_nb_model.fit(count_vec, dic_key_name)
    #
    # X_pred = count_vectorizer.transform(test_set["키워드"])
    #
    # total = accuracy_score(test_set["영역"], Y_pred)
    #
    # print("count_naive_bayes test set accuracy : ", total)
    #
    # print(classification_report(test_set["영역"], Y_pred, target_names=dic_key_name))
    #########################################

    #########################################
    # total = 0
    # c_perceptron_model = SGDClassifier(loss='perceptron')
    # c_perceptron_model.fit(tfidf_vec, dic_key_name)
    #
    # X_pred = count_vectorizer.transform(test_set["키워드"])
    # Y_pred = c_perceptron_model.predict(X_pred)
    # total = accuracy_score(test_set["영역"], Y_pred)
    #
    # print("count_perceptron test set accuracy : ", total)
    #
    # print(classification_report(test_set["영역"], Y_pred, target_names=dic_key_name))
    #########################################

    #########################################
    # lr_model = LogisticRegression(solver='liblinear', multi_class='auto')
    # lr_model.fit(count_vec.todense(), dic_key_name)
    # X_pred = count_vectorizer.transform(test_X)
    # Y_pred = lr_model.predict(X_pred)
    # total += accuracy_score(test_Y, Y_pred)
    # print("count svm test set accuracy : ", total)
    #########################################


def skipgrams(load_version):
    file = open('data/revision/revision_' + load_version + '.txt', 'r', encoding='utf-8', newline='\n')
    file_text = file.read()
    remove_num_doc = re.sub(r'\d', '', file_text)
    remove_eng_doc = re.sub('[a-zA-Z]', '', remove_num_doc)
    doc_split = remove_eng_doc.split()

    back_window = 2
    front_window = 2
    skipgram_counts = Counter()

    for idx, word in enumerate(doc_split):
        icw_min = max(0, idx - back_window)
        icw_max = min(len(doc_split) - 1, idx + front_window)
        icws = [ii for ii in range(icw_min, icw_max + 1) if ii != idx]
        for icw in icws:
            skipgram = (doc_split[idx], doc_split[icw])
            skipgram_counts[skipgram] += 1

    print('done')
    print('number of skipgrams: {}'.format(len(skipgram_counts)))
    # print('most common: {}'.format(skipgram_counts.most_common(100)))
    for word in skipgram_counts.most_common(100):
        print(word)


# n-gram
def word_ngram(num_gram, version):
    with open('data/revision/revision_' + version + '.txt', 'r', encoding='utf-8', newline='\n') as f:
        lines = f.readlines()

    temp_text = open('data/revision/revision_temp.txt', 'w', encoding='utf-8')

    for line in lines:
        temp_text.writelines(line.strip())
        temp_text.writelines("\n")

    temp_text.close()

    temp_text = open('data/revision/revision_temp.txt', 'r', encoding='utf-8')
    file = temp_text.read()

    sampling_file = ''.join(file.strip())
    # print(sampling_file)

    sample_file = sampling_file.replace('\n', ' ').replace('\r', ' ')

    ###### 영어와 숫자 제거 ######
    remove_num_doc = re.sub(r'\d', '', sample_file)
    remove_num_and_eng_doc = re.sub('[a-zA-Z]', '', remove_num_doc)
    # remove_num_and_eng_and_empty_doc = [x for x in remove_num_and_eng_doc if x]
    #######################

    text = remove_num_and_eng_doc.split(' ')
    ######### 숫자와 영문자 제거로 인해 생기는 공백 제거 #######
    text = tuple([x for x in text if x])
    ######################

    ngrams = [text[x:x + num_gram] for x in range(0, len(text))]

    return tuple(ngrams)


def check_duplication_word():
    # 다른 영역 키워드랑 겹치는 것이 있는지
    file = open('analysis/keyword_by_condition.txt', 'r', encoding='utf-8-sig', newline='\n')
    file_text = file.read()
    file_split = file_text.split("\n")
    corpus_list = []
    for i, line in enumerate(file_split):
        duplication = list(set(line.split(" ")))
        corpus_list.append(duplication)
        # tmp_line = ' '.join(duplication)

    for idx, one_corpus in enumerate(corpus_list):
        print("====================" + dic_key_name[idx] + "====================")
        for seq, one in enumerate(corpus_list):
            if idx == seq:
                continue
            print("=======" + dic_key_name[seq] + "=======")
            for word in one_corpus:
                if word in one:
                    print(word)

        for seq, word in enumerate(corpus_list[idx]):
            if word in one_corpus:
                print(word)

    # with open("analysis/duplication_check.txt", 'w', encoding='utf-8') as file:
    #     file.write(microsoft_stt_result)


# 한글이 깨지는 문제
def get_PMI():
    '''
        'bigram_chi_sq': nltk.collocations.BigramAssocMeasures().chi_sq,
        'trigram_chi_sq': nltk.collocations.TrigramAssocMeasures().chi_sq,
        'bigram_pmi': nltk.collocations.BigramAssocMeasures().pmi,
        'trigram_pmi': nltk.collocations.TrigramAssocMeasures().pmi,
        'bigram_raw_freq': nltk.collocations.BigramAssocMeasures().raw_freq,
        'trigram_raw_freq': nltk.collocations.BigramAssocMeasures().raw_freq,
        'bigram_student_t': nltk.collocations.BigramAssocMeasures().student_t,
        'trigram_student_t': nltk.collocations.TrigramAssocMeasures().student_t,
        'bigram_likelihood_ratio': nltk.collocations.BigramAssocMeasures().likelihood_ratio,
        'trigram_likelihood_ratio': nltk.collocations.TrigramAssocMeasures().likelihood_ratio,
        }.get(x, nltk.collocations.BigramAssocMeasures().pmi)
    '''
    file = open('analysis/keyword_by_condition.txt', 'r', encoding='utf-8-sig', newline='\n')
    file_text = file.read()

    measures = nltk.collocations.TrigramAssocMeasures()
    # measures = nltk.collocations.BigramAssocMeasures()

    finder = BigramCollocationFinder.from_words(file_text)
    # finder = TrigramCollocationFinder.from_words(word_tokenize(sampling_text))
    # print(finder.nbest(bigram_measures.pmi, 100))
    keywords = finder.score_ngrams(measures.pmi)
    # print(keywords)
    for x in keywords:
        print(x)
    for idx, i in enumerate(finder.score_ngrams(bigram_measures.pmi)):
    # for idx, i in enumerate(sorted(keywords., key=lambda x: x[1], reverse=True)):
    # for idx, i in enumerate(finder.nbest(bigram_measures.pmi, 5)):
        print(i)
        if idx == 1000:
            break


# Word2Vec 모델링 함수
# 첫 번째 파라미터 로드할 수정 데이터, 두 번째 파라미터는 저장할 모델 데이터 명
def W2V_modeling(load_name, save_name):

    # 자꾸 이미 모델링이 되어 있는데, main 함수에서 제거하는 것을 깜빡하니,
    # 모델링 파일 존재 유무를 먼저 파악함
    if os.path.isfile(save_name):
        print('"' + save_name + '" modeling file is exist already.\r\n')
        a = input('이미 모델링 파일이 존재, 모델링을 다시 진행? y or n\r\n')
        if a == 'y' or a == '':
            print('modeling start')
        else:
            print('기존 모델링 파일에서 유사어를 추출합니다.\r\n')
            return


    ############### 수정 버전별 W2V을 위한 수정 데이터 로드 #####################
    data = word2vec.LineSentence('resource/keyword_list.txt')
    ###############################

    ######################## 모델링 작업 ########################################
    model = word2vec.Word2Vec(data, size=100, window=100, min_count=1, hs=1, sg=1)   # 보통 차수는 50~100
    # 100차원 / 앞뒤 5개 / 최소 3번 / cbow = 0, skipgram = 1 / 학습 반복 횟수 iter = 100 /
    ################################

    ############################### Word2Vec 결과 저장 ##########################
    # model.save("datafirstclean_similar.model")     # 가공된 데이터 w2v 결과 저장
    # model.save("senten_separ.model")     # 문장단위 데이터 w2v 결과 저장
    model.save(save_name)
    ################################

    print("\r\nModeling is finished\r\n")


# 모델링 된 데이터 유사어 추출 함수
# 첫 번째 파라미터는 불러올 모델링 된 파일 명
# 두 번째 파라미터는 유사어 추출할 키워드 명
def model_similarity(load_file_name, keyword_name):
    print('==== "' + keyword_name + '" 키워드에 대한 유사어 추출 결과 ====')

    file = open('resource/keyword_list.txt', 'r', encoding='utf-8')
    frequency = {}

    all_word = file.read().split()
    for word in all_word:
        count = frequency.get(word, 0)
        frequency[word] = count + 1
    #############

    print(str(keyword_name) + " : " + str(frequency[keyword_name]))

    ###################### Word2Vec 결과 메모리에 로드#############################
    model = word2vec.Word2Vec.load(load_file_name)  # 문장 분리 형태소 X
    ###############################

    ################################################
    # index2word_set = set(model.wv.vocab.keys())
    # featureVec = np.zeros(model.vector_size, dtype="float32")
    # nwords = 0
    # for word in all_word:
    #     if word in index2word_set:
    #         featureVec = np.add(featureVec, model[word])
    #         nwords += 1.
    #     # Divide the result by the number of words to get the average
    # if nwords > 0:
    #     featureVec = np.divide(featureVec, nwords)
    # print(featureVec)
    ################################################

    ##############################################
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.preprocessing import LabelEncoder
    # label_encoder = LabelEncoder()
    #
    # label_encoder.fit(train_data.author)
    # train_data['author_id'] = label_encoder.transform(train_data.author)
    #
    # X_train_lda = np.array(list(map(np.array, train_data.lda_features)))
    # X_train_w2v = np.array(list(map(np.array, train_data.w2v_features)))
    # X_train_combined = np.append(X_train_lda, X_train_w2v, axis=1)
    #
    # # store all models in a dictionary
    # models = dict()
    #
    # lr = LogisticRegression()
    # param_grid = {'penalty': ['l1', 'l2']}
    #
    # best_lr_w2v = get_cross_validated_model(lr, param_grid, X_train_w2v, train_data.author_id)
    #
    # models['best_lr_w2v'] = best_lr_w2v
    ##############################################

    #################### Word2Vec 결과를 텍스트 형태 ################
    # model.wv.save_word2vec_format('vector_to_text.txt')
    ################################
    print(model[keyword_name])
    keyword_name = "위험한 신발"
    keyword_name = korean_pos_tokenize(keyword_name)
    keyword_name = ' '.join(keyword_name)

    similar_result = model.most_similar(positive=[keyword_name], topn=50)

    print(similar_result)
    print('\r\n')

    ######################################################
    # similar_result = model.wv.most_similar(positive=['', ''], negative=['', ''], topn=20)
    # model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
    # print(model.wv.doesnt_match("나".split()))
    # 유사 단어 관계 파악
    # model.wv.similarity('woman', 'man')
    ######################################################

    del model
    return similar_result


def get_cross_validated_model(model, param_grid, X, y, nr_folds=5):
    from sklearn.model_selection import GridSearchCV
    """ Trains a model by doing a grid search combined with cross validation.
    args:
        model: your model
        param_grid: dict of parameter values for the grid search
    returns:
        Model trained on entire dataset with hyperparameters chosen from best results in the grid search.
    """
    # train the model (since the evaluation is based on the logloss, we'll use neg_log_loss here)
    grid_cv = GridSearchCV(model, param_grid=param_grid, scoring='neg_log_loss', cv=nr_folds, n_jobs=-1, verbose=True)
    best_model = grid_cv.fit(X, y)
    # show top models with parameter values
    result_df = pd.DataFrame(best_model.cv_results_)
    show_columns = ['mean_test_score', 'mean_train_score', 'rank_test_score']
    for col in result_df.columns:
        if col.startswith('param_'):
            show_columns.append(col)
    print(result_df[show_columns].sort_values(by='rank_test_score').head())
    return best_model



def skipgrams(load_version):
    file = open('data/revision/revision_' + load_version + '.txt', 'r', encoding='utf-8', newline='\n')
    file_text = file.read()
    remove_num_doc = re.sub(r'\d', '', file_text)
    remove_eng_doc = re.sub('[a-zA-Z]', '', remove_num_doc)
    doc_split = remove_eng_doc.split()

    back_window = 2
    front_window = 2
    skipgram_counts = Counter()

    for idx, word in enumerate(doc_split):
        icw_min = max(0, idx - back_window)
        icw_max = min(len(doc_split) - 1, idx + front_window)
        icws = [ii for ii in range(icw_min, icw_max + 1) if ii != idx]
        for icw in icws:
            skipgram = (doc_split[idx], doc_split[icw])
            skipgram_counts[skipgram] += 1

    print('done')
    print('number of skipgrams: {}'.format(len(skipgram_counts)))
    # print('most common: {}'.format(skipgram_counts.most_common(100)))
    for word in skipgram_counts.most_common(100):
        print(word)


def kr_wordrank():
    file = open('exeltotxt.txt', 'r', encoding='utf-8', newline='\n')
    list_corpus = []
    for sentence in file:
        list_corpus.append(sentence.strip())
        # print(sentence)

    wordrank_extractor = KRWordRank(
        min_count=5,  # 단어의 최소 출현 빈도수 (그래프 생성 시)
        max_length=10,  # 단어의 최대 길이
        verbose=True
    )
    beta = 0.85  # PageRank의 decaying factor beta
    max_iter = 10
    keywords, rank, graph = wordrank_extractor.extract(list_corpus, beta, max_iter)

    for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:100]:
        print('%8s:\t%.4f' % (word, r))


def pipeline_func():
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score

    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report

    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                   ])

    lr = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('lr', SGDClassifier(loss='log')),
                   ])

    svm = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('svm', SGDClassifier(loss='hinge', penalty="l2")),
                   ])

    df = pd.read_excel('resource/영역_title_keyword_id_190819_notnan_space.xlsx')

    train_X, test_X, train_Y, test_Y = train_test_split(df["키워드"], df["영역"], test_size=0.1, random_state=42)

    svm.fit(train_X, train_Y)
    y_pred = svm.predict(test_X)
    print(y_pred)
    print(test_Y)

    print('accuracy %s' % accuracy_score(y_pred, test_Y))
    print(classification_report(test_Y, y_pred, target_names=dic_key_name))


def train_test_data_check():
    data_xls = pd.read_excel('resource/영역_title_keyword_id_190819_notnan_space.xlsx', 'Sheet1', index_col=None)
    train_set, test_set = split_set(data_xls, 0.1)
    print(len(train_set), "train +", len(test_set), "test")


''' 계층적 데이터 분할 함수 '''
def split_set(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == "__main__":
    start_time = time.time()

    ''' 엑셀 파일에서 텍스트 파일로 '''
    # exel_to_txt()

    ''' 영역에 따라서 키워드 분류 '''
    # keyword_extraction_by_condition()

    ''' sklearn 패키지를 사용하여 TF-IDF 및 countvec 특징 추출 '''
    Count_TFIDF_based_featuer_extracion()

    ''' TF-IDF 및 countvec 및 word2vec 을 통한 분류 '''
    Count_TFIDF_based_classification()

    ''' 중복된 키워드에 대한 가충치 설정 부분 '''
    # check_duplication_word()

    ''' PMI 필요 '''
    # get_PMI()

    ''' skipgram '''
    # skipgrams(load_save_txt_version)

    ''' 한국어 Text Rank '''
    # kr_wordrank()

    ''' Word2Vec 적용 후 분류 '''
    # W2V_modeling(0, "cress")
    # model_similarity('cress', '동요')
    # w2v_classification()

    ''' 학습용, 평가용 데이터 분할 '''
    # train_test_data_check()

    ''' 불용어 제거 '''
    # stopword_remove()

    e = int(time.time() - start_time)
    print('\r\n{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

