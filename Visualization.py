from gensim.models import word2vec
import pandas as pd
import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm

font_location = 'C:/Windows/Fonts/H2HDRM.ttf'  # For Windows
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dic_key_name = ["한글", "영어", "과학", "자연관찰", "수학", "영어동요", "동화", "사회관계", "예술경험", "건강안전", "동요"]


def model_similarity(load_file_name, keyword_name):
    print('==== "' + keyword_name + '" 키워드에 대한 유사어 추출 결과 ====')

    ########## 단어 빈도 카운트
    file = open('resource/keyword_list.txt', 'r', encoding='utf-8')
    frequency = {}
    all_word = file.read().split()
    for word in all_word:
        count = frequency.get(word, 0)
        frequency[word] = count + 1
    #############
    print(str(keyword_name) + " : " + str(frequency[keyword_name]))

    model = word2vec.Word2Vec.load(load_file_name) # 형태소 X
    ###############################

    ############################### Word2Vec 유사단어 결과 보기 ####################
    # try:
    #     # 해당 단어의 유사어가 존재, similar_result 리스트에 저장
    #     similar_result = model.most_similar(positive=[keyword_name], topn=50)
    #     # similar_result = model.most_similar(positive=[keyword_name, keyword_name2, keyword_name3], topn=50)
    # except KeyError:
    #     print("존재하지 않음")
    #     return 0
    ################################

    ####################################
    from sklearn.manifold import TSNE

    embedding_clusters = []
    word_clusters = []
    for word in dic_key_name:
        embeddings = []
        words = []
        for similar_word, _ in model.most_similar(word, topn=30):
            words.append(similar_word)
            embeddings.append(model[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    tsne_plot_similar_words('Similar words group vector from CRES data ', dic_key_name, embeddings_en_2d, word_clusters, 0.7,
                            'similar_words_cres.png')
    ####################################
    del model
    # return similar_result


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors
