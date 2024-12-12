import nltk
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases, LdaModel
from gensim.corpora import Dictionary
import pyLDAvis.gensim
import pyLDAvis
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def preprocess_text(df, with_certain_words_removal = False, data_is_big = False):

    texts = df['text'].astype(str).tolist()
    print('Tokenization starting ---')

    # Lower-caseing and tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    texts = [tokenizer.tokenize(text.lower()) for text in texts]

    print('Tokenization done')
    #print(texts[:3])

    # Removing numbers 
    texts =[[w for w in text if not w.isnumeric()] for text in texts]

    print('Numbers removed')
    #print(texts[:3])

    # Removing 2-letter words
    texts = [[w for w in text if len(w) > 2] for text in texts]

    print('Two letter words removed')
    #print(texts[:3])

    # Removing written-out numbers
    written_numbers = {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", 
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", 
        "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", 
        "eighty", "ninety", "hundred", "thousand", "million", "billion"
    }
    texts = [[w for w in text if w not in written_numbers] for text in texts]

    print('Written-out numbers removed')
    #print(texts[:3])

    if data_is_big:
        common_verbs = set([
        'be', 'have', 'do', 'say', 'make', 'go', 'know', 'take', 'see', 'get', 
        'give', 'come', 'think', 'look', 'want', 'use', 'find', 'tell', 'ask', 
        'work', 'seem', 'feel', 'try', 'leave', 'call', 'need', 'become', 'put', 
        'mean', 'keep', 'let', 'begin', 'seem', 'help', 'talk', 'turn', 'start', 
        'show', 'hear', 'play', 'run', 'move', 'live', 'believe', 'hold', 'bring', 'wear', 'come', 'use', 'work'
        ])


        # Removing verbs
        texts = [
            [word for word in text if word not in common_verbs]  
            for text in texts
        ]

        print('Verbs removed BIG')
    else: 
        texts = [
        [word for word, tag in pos_tag(word_tokenize(' '.join(text))) if tag[:2] != 'VB']  # Removing verbs
        for text in texts
        ]

        print('Verbs removed SMALL')



    # Lematization
    lemmatizer = WordNetLemmatizer()
    #texts = [[lemmatizer.lemmatize(w) for w in text] for text in texts]
    texts = [[lemmatizer.lemmatize(w, pos='v') for w in text] for text in texts]  
    texts = [[lemmatizer.lemmatize(w, pos='n') for w in text] for text in texts] 

    print('Lematization done')
    #print(texts[:3])

    # Stopwords removal
    stop_words = set(stopwords.words('english'))  
    texts = [[w for w in text if w not in stop_words] for text in texts]

    print('Stopwords removed')
    #print(texts[:3])

    # Expand as needed, idk if it makes any sense
    if with_certain_words_removal:
        product_specific_nouns = set([
            'picture', 'photo', 'jacket', 'dress', 'shoe', 'shirt', 'jeans', 'skirt', 
            'blouse', 'pant', 'hat', 'sock', 'bag', 'watch', 'get', 'wear' , 'like', 'watch'
        ])
        texts = [[w for w in text if w not in product_specific_nouns] for text in texts]
        print('Choosed words removed')

    # Finding bigrams (pairs of words which come together, like data science, water bottle)
    bigram = Phrases(texts, min_count=30)
    texts = [[w for w in text] + [w for w in bigram[text] if '_' in w] for text in texts]

    print('Bigrams done')
    #print(texts[:3])

    # # Dictionary creation and filtering rare and common words
    dictionary = Dictionary(texts)
    # dictionary.filter_extremes(no_below=100, no_above=0.8)

    # print('Common and rare words removed')
    # #print(texts[:3])

    # Bag-of-words representation
    texts_bow = [dictionary.doc2bow(text) for text in texts]
    print(texts_bow[:3])

    id2token = {id: token for token, id in dictionary.token2id.items()}

    print('Preprocessing done')
    #print(texts[:3])


    return texts_bow, dictionary, id2token

def display_visuals_LDA(model, texts_bow, dictionary):
    LDAvis_prepared = pyLDAvis.gensim.prepare(model, texts_bow, dictionary)
    return LDAvis_prepared

def display_top_words_for_topics(lda_model, n_topics, n_words=10):
    for topic_id in range(n_topics):
        top_words = lda_model.show_topic(topic_id, topn=n_words)
        words = [word for word, _ in top_words]  
        print(f"Topic {topic_id}: {', '.join(words)}")



def LDA_training(df,with_certain_words_removal = False, n_topics = 5, chunksize = 1000, passes = 100, iterations = 200, update_every = 1, eval_every=float('inf'), texts_bow=None, dictionary = None, id2word = None):
    
    if texts_bow is None or dictionary is None or id2word is None:
        texts_bow, dictionary, id2word = preprocess_text(df, with_certain_words_removal=with_certain_words_removal)

    print('--- Model starting ---')
    model = LdaModel(corpus=texts_bow, id2word=id2word, chunksize=chunksize, 
                       alpha='auto', eta='auto', 
                       iterations=iterations, num_topics=n_topics, 
                       passes=passes, update_every=update_every, eval_every=eval_every)   


    top_topics = model.top_topics(texts_bow, topn=3)

    # This is how we evaluate lda model
    avg_topic_coherence = sum([t[1] for t in top_topics]) / n_topics
    print('Średnia koherencja tematów: %.4f.' % avg_topic_coherence)

    # Displaying top words for topic
    display_top_words_for_topics(model, n_topics=n_topics)


    return model,texts_bow, dictionary



def classification(df, lda_model, texts_bow):
    print('--Classification starting---')
    
    topics_per_document = [lda_model.get_document_topics(bow) for bow in texts_bow]
    
    assigned_topics = []
    for doc_topics in topics_per_document:
        if doc_topics:
            assigned_topics.append(max(doc_topics, key=lambda x: x[1])[0])  
        else:
            assigned_topics.append(None) 
    
    df['assigned_topic'] = assigned_topics
    return df

def add_top_words_to_df(df, lda_model, n_topics, n_words=10):
    top_words_per_topic = {}
    
    for topic_id in range(n_topics):
        top_words = lda_model.show_topic(topic_id, topn=n_words)
        words = [word for word, _ in top_words]
        top_words_per_topic[topic_id] = ', '.join(words)
    
    df['top_words'] = df['assigned_topic'].map(top_words_per_topic)
    return df

def prepare_data_for_svm(df, vectorizer=None):
    if vectorizer is None:
        vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(df['text'])
    y = df['assigned_topic']

    return X, y, vectorizer


def train_svm_classifier(X_train, y_train):
    svm = SVC(kernel='linear', C=1, random_state=42, class_weight='balanced')
    svm.fit(X_train, y_train)
    return svm

def lda_and_svm_pipeline(df, model, texts_bow, with_certain_words_removal=False, n_topics=15, test_size=0.2, random_state=42):

    print('---Begining SVM---')
    df_with_topics = classification(df, model, texts_bow)
    df_with_topics_and_words = add_top_words_to_df(df_with_topics, model, n_topics)
    X, y, vectorizer = prepare_data_for_svm(df_with_topics_and_words)

    print('---Train test splitting---')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print('---SVM training---')
    svm = train_svm_classifier(X_train, y_train)

    print('---SVM predicting---')
    y_pred = svm.predict(X_test)

    print("---Classification Report:---\n")
    print(classification_report(y_test, y_pred))

    return svm, vectorizer, df_with_topics_and_words



def create_review_topic_matrix(df, lda_model, texts_bow, n_topics):

    unique_reviews = sorted(df['rating'].unique())
    
    matrix = np.zeros((len(unique_reviews), n_topics))

    topics_per_document = [lda_model.get_document_topics(bow, minimum_probability=0.0) for bow in texts_bow]

    for i, review in enumerate(unique_reviews):
        indices = df[df['rating'] == review].index.tolist()

        for idx in indices:
            if idx < len(topics_per_document):  
                topic_probs = topics_per_document[idx]
                for topic_id, prob in topic_probs:
                    matrix[i, topic_id] += prob

        row_sum = matrix[i, :].sum()
        if row_sum > 0:
            matrix[i, :] /= row_sum

    matrix = pd.DataFrame(matrix, index=unique_reviews, columns=[f"Topic_{i}" for i in range(n_topics)])

    plt.figure(figsize=(20, 6))
    sns.heatmap(matrix, annot=True, fmt=".5f", cmap="coolwarm", cbar=True)
    plt.title("Review-Topic Matrix Heatmap")
    plt.xlabel("Topics")
    plt.ylabel("Review Scores")
    plt.show()

    return matrix




def generate_topic_rating_matrix(df, n_topics=15):
  
    matrix = np.zeros((5, n_topics))  
    
    for rating in range(1, 6): 
        subset = df[df['rating'] == rating]
        
        for topic in range(n_topics):
            topic_assignments = subset[subset['assigned_topic'] == topic]
            if len(topic_assignments) > 0:
                matrix[rating - 1, topic] = topic_assignments.shape[0] / len(subset)  
    
    matrix_df = pd.DataFrame(matrix, columns=[f"Topic_{i}" for i in range(n_topics)], index=[1, 2, 3, 4, 5])

    plt.figure(figsize=(20, 6))
    sns.heatmap(matrix_df, annot=True, fmt=".5f", cmap="coolwarm", cbar=True)
    plt.title("Review-Topic Matrix Heatmap")
    plt.xlabel("Topics")
    plt.ylabel("Review Scores")
    plt.show()

    return matrix_df



