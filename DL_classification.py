from sklearn.metrics import confusion_matrix
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings('ignore')

dic_key_name = ["한글", "영어", "과학", "자연관찰", "수학", "영어동요", "동화", "사회관계", "예술경험", "건강안전", "동요"]
dic_key_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def neural_net_pytorch():
    # train_X, test_X, train_Y, test_Y = train_test_split(tfidf, np.array(dic_key_num), test_size=0.1, random_state=42)
    # # train_X, test_X, train_Y, test_Y = train_test_split(tfidf.toarray(), np.array(dic_key_num), test_size=0.1, random_state=42)

    # 데이터 형식 fitting
    train_X = torch.from_numpy(tfidf_vec.toarray()).float()
    train_Y = torch.from_numpy(np.array(dic_key_num)).long()

    X_pred = tfidf_vectorizer.transform(test_set["키워드"])

    test_X = torch.from_numpy(X_pred.toarray()).float()
    test_Y = torch.from_numpy(test_set["영역"].values).float()

    train = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train, batch_size=100, shuffle=True)

    # 인스턴스 생성
    model = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    for epoch in range(100):
        total_loss = 0

        for train_x, train_y in train_loader:
            train_x, train_y = Variable(train_x), Variable(train_y)
            optimizer.zero_grad()
            output = model(train_x)
            loss = criterion(output, train_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()

        # print(epoch + 1, total_loss)

    test_x, test_y = Variable(test_X), Variable(test_Y)
    result = torch.max(model(test_x).data, 1)[1]
    accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())
    print(accuracy)


# 신경망 구성
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(train_X.shape[1], 128)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(256, 256)
        # self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 11)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc5(x)
        return F.log_softmax(x)


def split_set(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]


def TFIDF(X_train, X_test,MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print("tf-idf with",str(np.array(X_train).shape[1]),"features")
    return (X_train,X_test)


def Build_Model_DNN_Text(shape, nClasses, dropout=0.5):
    """
    buildModel_DNN_Tex(shape, nClasses,dropout)
    Build Deep neural networks Model for text classification
    Shape is input feature space
    nClasses is number of classes
    """
    model = Sequential()
    node = 512      # number of nodes
    nLayers = 4     # number of  hidden layer
    model.add(Dense(node, input_dim=shape, activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0, nLayers):
        model.add(Dense(node, input_dim=node, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


from sklearn.metrics import accuracy_score

def deep_NN():
    X_train_tfidf = tfidf_vec.toarray()
    X_test_tfidf = tfidf_vectorizer.transform(test_set["키워드"]).toarray()

    model_DNN = Build_Model_DNN_Text(X_train_tfidf.shape[1], 70)
    print(model_DNN.summary())
    model_DNN.fit(X_train_tfidf, dic_key_num,
                                  validation_data=(X_test_tfidf, test_set["영역"]),
                                  epochs=70,
                                  batch_size=128,
                                  verbose=0)
    predicted = model_DNN.predict(X_test_tfidf)
    print(predicted.shape)
    total = accuracy_score(test_set["영역"], predicted.argmax(axis=1))
    print("tfidf_dnn test set accuracy : ", total)

    print(metrics.classification_report(test_set["영역"], predicted.argmax(axis=1)))
    conf_mx = confusion_matrix(test_set["영역"], predicted.argmax(axis=1))
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()

    # predicted = model_DNN.predict_proba(X_test_tfidf)
    # fpr, tpr, _ = metrics.roc_curve(test_set["영역"], predicted.argmax(axis=1))
    # auc = metrics.roc_auc_score(test_set["영역"], predicted.argmax(axis=1))
    # plt.plot(fpr, tpr, label="AUC="+str(auc))
    # plt.legend(loc=4)
    # plt.show()


from keras.layers import Dropout, Dense,Input,Embedding,Flatten, MaxPooling1D, Conv1D
from keras.models import Sequential,Model
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
from keras.layers.merge import Concatenate

def Build_Model_CNN_Text(word_index, embeddings_index, nclasses, MAX_SEQUENCE_LENGTH=500, EMBEDDING_DIM=50, dropout=0.5):

    """
        def buildModel_CNN(word_index, embeddings_index, nclasses, MAX_SEQUENCE_LENGTH=500, EMBEDDING_DIM=50, dropout=0.5):
        word_index in word index ,
        embeddings_index is embeddings index, look at data_helper.py
        nClasses is number of classes,
        MAX_SEQUENCE_LENGTH is maximum lenght of text sequences,
    """

    model = Sequential()
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) !=len(embedding_vector):
                print("could not broadcast input array from shape",str(len(embedding_matrix[i])),
                                 "into shape",str(len(embedding_vector))," Please make sure your"
                                 " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    # applying a more complex convolutional approach
    convs = []
    filter_sizes = []
    layer = 5
    print("Filter  ",layer)
    for fl in range(0,layer):
        filter_sizes.append((fl+2))
    node = 128
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    for fsz in filter_sizes:
        l_conv = Conv1D(node, kernel_size=fsz, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        #l_pool = Dropout(0.25)(l_pool)
        convs.append(l_pool)
    l_merge = Concatenate(axis=1)(convs)
    l_cov1 = Conv1D(node, 5, activation='relu')(l_merge)
    l_cov1 = Dropout(dropout)(l_cov1)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(node, 5, activation='relu')(l_pool1)
    l_cov2 = Dropout(dropout)(l_cov2)
    l_pool2 = MaxPooling1D(30)(l_cov2)
    l_flat = Flatten()(l_pool2)
    l_dense = Dense(1024, activation='relu')(l_flat)
    l_dense = Dropout(dropout)(l_dense)
    l_dense = Dense(512, activation='relu')(l_dense)
    l_dense = Dropout(dropout)(l_dense)
    preds = Dense(nclasses, activation='softmax')(l_dense)
    model = Model(sequence_input, preds)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


