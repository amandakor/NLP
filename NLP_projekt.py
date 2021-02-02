#IMPORT
import csv
from nltk.tokenize import word_tokenize 
import string
from nltk.corpus import stopwords
import nltk
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random

from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.autograd as autograd
import torchtext

#pobieramy nasze dane i zapisujemy je do krotki
with open('movies_data.csv', encoding="utf-8") as f:  # Or whatever encoding is right
    reader = csv.reader(f)
    data = [tuple(row) for row in reader]
    data = data[1:]
    
movie_revs = []
for elem in data:

    tokens = word_tokenize(elem[0])
    tokens = [w.lower() for w in tokens]
    # usuwamy znaki interpunkcyjne
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # usuwamy tokeny, ktore nie sa literami
    words = [word for word in stripped if word.isalpha()]
    #stopwords
    
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    words = [elem for elem in words if not elem in ("br", "nt", "f")]
    movie_revs.append((words, elem[1]))

# czy dane są zbalansowane?
howmuch_revs = []
for i in range(len(movie_revs)):
    howmuch_revs.append(len(movie_revs[i][0]))
print('Jaki jest rozklad wszystkich recenzji:\n',howmuch_revs[:50])

#liczba pozytywnych i negatywnych recenzji

pos = []
neg = []
for elem in movie_revs:
    if elem[1] == '1':
        pos.append(elem[0])
    if elem[1] == '0':
        neg.append(elem[0])

print('Pozytywnych recenzji jest: ', len(pos))
print('Negatywnych recenzji jest: ', len(neg))
      
#ile wystapiło wszystkich slow a ile unikatowych?

all_w = pos + neg
#słowa unikatowe, musze pozbyc sie listy w liscie --> unhashable 
all_words = []
for sublist in all_w:
    for item in sublist:
        all_words.append(item)

print('Wszystkich slow jest:\n', len(all_words))
print('Liczba slow unikatowych:\n',len(set(all_words)))


#frekwencja slow
fdist = FreqDist(all_words)
fdist.N()
fdist.plot(40, title='Rozklad wystepowania 40 najpopularniejszych tokenow')

#wordcloud
cloud = WordCloud(max_font_size=80,colormap="hsv").generate_from_frequencies(fdist)
plt.figure(figsize=(16,12)) #wymiar obrazka
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#Naive Bayes, Logistic Regression, Support Vector Machine

word_features = [x[0] for x in fdist.most_common(500)]

#Funkcja zwracająca slownik, gdzie klucze to word_features, a klucze to True lub False w zależnosci od tego czy dane slowo występuje w dokumencie czy nie
def find_features(document): 
    words = set(document)    
    features = {}            
    for w in word_features: 
        features[w] = (w in words) 
    return features

#wywoluje te funkcje dla wszystkich dokumentow, dodatkowo zapisuje do krotki informacje o kategorii recenzji
featuresets = [(find_features(rev),category) for (rev,category) in movie_revs]

#zbiorcza klasyfikacja oparta o wiele klasyfikatorow

class AggClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    
#listy do zbierania danych z 10-krotnej budowy modelu
NB_class = [] #Naive Bayes
LR_class = [] #regresja logistyczna
SVM_class = [] #svm
agg_class = [] # klasyfikator zbiorczy

for i in range(11):
    random.shuffle(featuresets)
    #### Dane do trenowania i testowania modelu 80% do trenowania i 20% do testowania
    training_set = featuresets[:40000] 
    testing_set = featuresets[40000:50000]  
    #NB
    NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
    NB_class.append(nltk.classify.accuracy(NB_classifier,testing_set)*100)
    #LR
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    LR_class.append(nltk.classify.accuracy(LogisticRegression_classifier, testing_set)*100)
    #SVM
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    SVM_class.append(nltk.classify.accuracy(LinearSVC_classifier, testing_set)*100)
    #AGG
    agg_classifier = AggClassifier(NB_classifier, LogisticRegression_classifier, LinearSVC_classifier)
    agg_class.append(nltk.classify.accuracy(agg_classifier, testing_set)*100)
    
print("Srednia dokladnosc NB wynosi:\n", np.mean(NB_class))
print("Srednia dokladnosc LR wynosi:\n", np.mean(LR_class))
print("Srednia dokladnosc SVM wynosi:\n", np.mean(SVM_class))
print("Srednia dokladnosc AGG wynosi:\n", np.mean(agg_class))

#Bag of Words

random.shuffle(movie_revs)
#dane do trenowania
data = movie_revs[:40000]
#dane do testowania
test_data = movie_revs[40000:50000]
#Etykiety
label_to_ix = {'0': 0, '1': 1}

word_to_ix = {} #tutaj wrzucamy wszystkie slowa, kazde ma indywidualny numer (kolejna liczba naturalna)
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
            
VOCAB_SIZE = len(word_to_ix)  #ile wszysktich slow --> 133252
NUM_LABELS = len(label_to_ix) #ile kategorii --> 2

#Pomocnicze funkcje

def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)

def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])  #https://pytorch.org/docs/stable/tensors.html

#Model

class BoWClassifier(nn.Module):  
    def __init__(self, num_labels, vocab_size):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_labels)
        
    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim = 1)
 
    
model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

n_iters = 100
for epoch in range(n_iters):
    for instance, label in data:     
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        target = autograd.Variable(make_target(label, label_to_ix))
    
        #forward
        log_probs = model(bow_vec)
        loss = loss_function(log_probs, target)
        
        #backward
        loss.backward()
        optimizer.step()
        
        #zerujemy gradient
        optimizer.zero_grad()
        
for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print(log_probs)
    
list(model.parameters())

#GloVe

glove = torchtext.vocab.GloVe(name="6B", dim=200)

def get_rev_vectors(glove_vector):
    train, valid, test = [], [], [] #tworze trzy listy na dane train, valid i test
    for i, line in enumerate(movie_revs): #przechodze dane 
        rev = line[0]                  #recenzja to pierwszy element 
        rev_emb = sum(glove_vector[w] for w in rev) #wektor recenzji będzie sumą embeddingow slow w niego wchodzących
        label = torch.tensor(int(line[-1] == "1")).long() #generuje dwa rodzaje labelow: "1" - gdy pos, "0" gdy negative
            
        #dzielimy dane na trzy kategorie
        if i < 35000:                  
            train.append((rev_emb, label)) # 70% danych treningowych
        elif i >= 35000 and i < 42500:
            valid.append((rev_emb, label)) # 15% danych do walidacji
        else:           
            test.append((rev_emb, label)) # 15% danych testowych
    return train, valid, test


train, valid, test = get_rev_vectors(glove) #przygotowanie danych w oparciu o glove

# losowe tasowanie słow (300) dla train valid i test
train_loader = torch.utils.data.DataLoader(train, batch_size=300, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=300, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=300, shuffle=True)


def train_network(model, train_loader, valid_loader, num_epochs=5, learning_rate=1e-5):
    criterion = nn.CrossEntropyLoss()  #funkcja kosztu
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #optymalizator ADAM
    losses, train_acc, valid_acc = [], [], []
    epochs = []
    for epoch in range(num_epochs):          #dla kazdej epoki
        for revs, labels in train_loader:  #przechodze dane treningowe
            optimizer.zero_grad()
            pred = model(revs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        losses.append(float(loss))           #zapisuje wartosc funkcji kosztu
        if epoch % 20 == 0:                   #co 20 epoke 
            epochs.append(epoch)             #zapisz numer epoki
            train_acc.append(get_accuracy(model, train_loader))   #dokladnosc na zbiorze testowym
            valid_acc.append(get_accuracy(model, valid_loader))   #dokladnosc na zbiorze treningowym
            print(f'Epoch number: {epoch+1} | Loss value: {loss} | Train accuracy: {round(train_acc[-1],3)} | Valid accuracy: {round(valid_acc[-1],3)}')
    #Rysowanie wykresow
    plt.title("Training Curve")
    plt.plot(losses, label="Train dataset")
    plt.xlabel("Epoch number")
    plt.ylabel("Loss value")
    plt.show()

    plt.title("Training Curve")
    plt.plot(epochs, train_acc, label="Train dataset")
    plt.plot(epochs, valid_acc, label="Validation dataset")
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

#Funkcja wyznaczająca dokładność predykcji:
def get_accuracy(model, data_loader):
    correct, total = 0, 0  #ile ok, ile wszystkich
    for revs, labels in data_loader: #przechodzi dane
        output = model(revs)         #jak dziala model
        pred = output.max(1, keepdim=True)[1]  #ktora kategoria
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.shape[0]
    return correct / total


#uproszczony zapis sieci neuronowej
mymodel = nn.Sequential(nn.Linear(200, 100),  #przekształcenie liniowe R^200 ---> R^100
                        nn.ReLU(),          #wyniki przekształcam funkcją ReLU
                        nn.Linear(100, 50),  #kolejne przekształcenie liniowe R^100--->R^50
                        nn.ReLU(),          #znow ReLU
                        nn.Linear(50, 2))   #znow przekształcenie liniowe

train_network(mymodel, train_loader, valid_loader, num_epochs=100, learning_rate=0.0001)

print("Final test accuracy:", get_accuracy(mymodel, test_loader)) #dokladnosc na zbiorze testowym


#TBatcher

def get_rev_words(glove_vector, data): #argumenty: glove_vector - info o embeddingach, data = dane
    train, valid, test = [], [], []  #puste listy na dane treningowe, walidacyjne, do testowania
    for i, line in enumerate(movie_revs):
        rev = line[0]        #tutaj znajduje sie dana recenzja - slowa z niej
        idx = [glove_vector.stoi[w] for w in rev if w in glove_vector.stoi] #zapisuje indeksy slow ktore mialy embeddingi
        if not idx: #jezeli zdarzy sie recenzja bez zadnego embeddingu to pobiera kolejny rekord
            continue
        idx = torch.tensor(idx) #zapisuje indeksy jakot tensor
        label = torch.tensor(int(line[-1] == "1")).long() #label dla recenzji: 0 dla "0" - neg lub 1 dla "1" - pos
        #chce sobie podzielic dane na trzy kategorie: zbior treningowy, walidacyjny i testowy 
        if i < 35000:  #70% danych
            train.append((idx, label))
        elif i >= 35000 and i < 42500: #15 % danych
            valid.append((idx, label))
        else:            # pozostale 15% danych
            test.append((idx, label))
    return train, valid, test #zwraca zbior treningowy, walidacyjny i testowy

train, valid, test = get_rev_words(glove, movie_revs) #generuje listy

#Problem z batchingiem, wyrownanie tensorow dla recenzji

class TBatcher:
    def __init__(self, revs, batch_size=32, drop_last=False):
        self.revs_by_length = {} #slownik, przechowuje klucze - dlugosci i wartosci - liste tweetow o zadanej dlugosci
        for words, label in revs:
            wlen = words.shape[0] #liczy dlugosc recenzji (ile ma embeddingow)
            
            if wlen not in self.revs_by_length: #jak jeszcze nie pojawila sie recenzja o takiej dlugosci
                self.revs_by_length[wlen] = []  #to stworz ja i przypisz jej pustą liste
                
            self.revs_by_length[wlen].append((words, label),) #dodaje do listy krotke slowa, label
         
        #tworze DataLoader dla kazdego zbioru tweetow o tej samej dlugosci
        self.loaders = {wlen : torch.utils.data.DataLoader(revs, batch_size=batch_size, shuffle=True, drop_last=drop_last) for wlen, revs in self.revs_by_length.items()}
    
    #Iterator, to nie takie wazne...
    def __iter__(self): 
        iters = [iter(loader) for loader in self.loaders.values()] #tworze iterator dla kazdej dlugosci tweetow
        while iters:
            im = random.choice(iters) #generuje losowy iterator
            try:
                yield next(im)      #yield uzywamy kiedy iterujemy po sekwencji ale nie chcemy przechowywac calej sekwencji w pamieci (cos jak return)
            except StopIteration:
                iters.remove(im)
                
#LSTM 

lstm_layer = nn.LSTM(input_size=200,   #wymiar wejscia - bo mam embeddingi 200D
                    hidden_size=5,   #wymiar cech w stanie ukrytym
                    batch_first=True)


class T_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(glove.vectors) #embeddingi
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True) #LSTM
        self.fc = nn.Linear(hidden_size, num_classes)  #przeksztalcenie liniowe
    
    def forward(self, x):
        x = self.emb(x)
        h0 = torch.zeros(1, x.size(0), self.hidden_size) #początkowy  h0
        c0 = torch.zeros(1, x.size(0), self.hidden_size) #początkowy c0
        out, _ = self.lstm(x, (h0, c0))  #LSTM
        out = self.fc(out[:, -1, :]) #przeksztlcam jeszcze liniowo ostatni output
        return out

#funkcja do treningu

def LSTM_train(model, train, valid, num_epochs=5, learning_rate=1e-5):
    criterion = nn.CrossEntropyLoss() #funkcja kosztu
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #optymalizator modelu
    losses, train_acc, valid_acc, epochs = [], [], [], []  #cztery listy na wartosci funkcji kosztu, dokladnosc na zbiorze testowym i walidacyjnym, numer epoki
    
    for epoch in range(num_epochs): #przechodz kolejne epoki (iteracje)
        for revs, labels in train: #przechodzi dane ze zbioru testowego
            optimizer.zero_grad()   #zerowanie gradientu
            pred = model(revs)    #co mowi model?
            loss = criterion(pred, labels)   #wartosc funkcji kosztu - porownanie tego co mowi model, a tego jak jest
            loss.backward()                  #pochodna po funkcji kosztu
            optimizer.step()                 #aktualizacja parametrow
        losses.append(float(loss))           #zapisz aktualną wartosc funkcji kosztu
        epochs.append(epoch)                 #zapisz aktualny numer epoki
        train_acc.append(get_accuracy(model, train_loader))   #dokladnosc na zbiorze testowym
        valid_acc.append(get_accuracy(model, valid_loader))   #dokladnosc na zbiorze treningowym
        print(f'Epoch number: {epoch+1} | Loss value: {loss} | Train accuracy: {round(train_acc[-1],3)} | Valid accuracy: {round(valid_acc[-1],3)}')
 
    #Rysowanie wykresow
    plt.title("Training Curve")
    plt.plot(losses, label="Train dataset")
    plt.xlabel("Epoch number")
    plt.ylabel("Loss value")
    plt.show()

    plt.title("Training Curve")
    plt.plot(epochs, train_acc, label="Train dataset")
    plt.plot(epochs, valid_acc, label="Validation dataset")
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    
test_loader = TBatcher(test, batch_size=64, drop_last=False)
train_loader = TBatcher(train, batch_size=64, drop_last=True)  #dane treningowe z batchem
valid_loader = TBatcher(valid, batch_size=64, drop_last=False)  #dane walidacyjne z batchem


lstm_model = T_LSTM(200, 50, 2) #model
LSTM_train(lstm_model, train_loader, valid_loader, num_epochs=15, learning_rate=2e-4) #trenuje
print(get_accuracy(lstm_model, test_loader)) #dokladnosc na zbiorze testowym


#GRU

gru_layer = nn.GRU(input_size=200,   #wymiar wejscia 200D
                   hidden_size=5,   #wymiar cech w stanie ukrytym
                   batch_first=True) 

class T_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(glove.vectors) #embeddingi
        self.hidden_size = hidden_size 
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)  #GRU
        self.fc = nn.Linear(hidden_size, num_classes)   #przeksztalcenie liniowe
    
    def forward(self, x):
        x = self.emb(x)  #embeddingi
        h0 = torch.zeros(1, x.size(0), self.hidden_size) #początkowy stan ukryty
        out, _ = self.gru(x, h0)   #GRU
        out = self.fc(out[:, -1, :]) #ostatni output przeksztalcamy liniowo jeszcze
        return out
    
model_gru = T_GRU(200, 50, 2) #buduje model
LSTM_train(model_gru, train_loader, valid_loader, num_epochs=15, learning_rate=2e-4) #trenuje, biore funkcje do treningu z LSTM :)

print(get_accuracy(model_gru, test_loader)) #dokladnosc na zbiorze testowym
