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




def preprocess_text(df, with_certain_words_removal = False):

    texts = df['text'].astype(str).tolist()
    print('Tokenization starting ---')

    # Lower-caseing and tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    texts = [tokenizer.tokenize(text.lower()) for text in texts]

    print('Tokenization done')
    print(texts[:3])

    # Removing numbers 
    texts =[[w for w in text if not w.isnumeric()] for text in texts]

    print('Numbers removed')
    print(texts[:3])

    # Removing one-letter words
    texts = [[w for w in text if len(w) > 1] for text in texts]

    print('One letter words removed')
    print(texts[:3])

    # Removing written-out numbers
    written_numbers = {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", 
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", 
        "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", 
        "eighty", "ninety", "hundred", "thousand", "million", "billion"
    }
    texts = [[w for w in text if w not in written_numbers] for text in texts]

    print('Written-out numbers removed')
    print(texts[:3])

    # Lematization
    lemmatizer = WordNetLemmatizer()
    #texts = [[lemmatizer.lemmatize(w) for w in text] for text in texts]
    texts = [[lemmatizer.lemmatize(w, pos='v') for w in text] for text in texts]  
    texts = [[lemmatizer.lemmatize(w, pos='n') for w in text] for text in texts] 

    print('Lematization done')
    print(texts[:3])

    # Stopwords removal
    stop_words = set(stopwords.words('english'))  
    texts = [[w for w in text if w not in stop_words] for text in texts]

    print('Stopwords removed')
    print(texts[:3])

    # Expand as needed, idk if it makes any sense
    if with_certain_words_removal:
        product_specific_nouns = set([
            'picture', 'photo', 'jacket', 'dress', 'shoe', 'shirt', 'jeans', 'skirt', 
            'blouse', 'pant', 'hat', 'sock', 'bag', 'watch', 'get', 'wear' , 'like'
        ])
        texts = [[w for w in text if w not in product_specific_nouns] for text in texts]
        print('Choosed words removed')

    # Finding bigrams (pairs of words which come together, like data science, water bottle)
    bigram = Phrases(texts, min_count=30)
    texts = [[w for w in text] + [w for w in bigram[text] if '_' in w] for text in texts]

    print('Bigrams done')
    print(texts[:3])

    # Dictionary creation and filtering rare and common words
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=3, no_above=0.7)

    print('Common and rare words removed')
    print(texts[:3])

    # Bag-of-words representation
    texts_bow = [dictionary.doc2bow(text) for text in texts]
    print(texts_bow[:3])

    id2token = {id: token for token, id in dictionary.token2id.items()}

    print('Preprocessing done')
    print(texts[:3])


    return texts_bow, dictionary, id2token

def LDA_training(df,with_certain_words_removal = False, n_topics = 5, chunksize = 1000, passes = 1, iterations = 100, eval_every = None):
    texts_bow, dictionary, id2word = preprocess_text(df, with_certain_words_removal=with_certain_words_removal)

    print('--- Model starting ---')
    model = LdaModel(corpus=texts_bow, id2word=id2word, chunksize=chunksize, 
                       alpha='auto', eta='auto', 
                       iterations=iterations, num_topics=n_topics, 
                       passes=passes, eval_every=eval_every)

    top_topics = model.top_topics(texts_bow, topn=3)


    avg_topic_coherence = sum([t[1] for t in top_topics]) / n_topics
    print('Średnia koherencja tematów: %.4f.' % avg_topic_coherence)

    print(top_topics)

    # Visuals
    # LDAvis_prepared = pyLDAvis.gensim.prepare(model, texts_bow, dictionary)
    # pyLDAvis.display(LDAvis_prepared)

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

    df['text'] = df['text'].fillna('')

    X = vectorizer.fit_transform(df['text'])
    y = df['assigned_topic']

    return X, y, vectorizer


def train_svm_classifier(X_train, y_train):
    svm = SVC(kernel='linear', C=1, random_state=42, class_weight='balanced')
    svm.fit(X_train, y_train)
    return svm

def lda_and_svm_pipeline(df, model, texts_bow, with_certain_words_removal=False, n_topics=15, test_size=0.2, random_state=42):
    df_with_topics = classification(df, model, texts_bow)
    df_with_topics_and_words = add_top_words_to_df(df_with_topics, model, n_topics)

    print('Preparing data for SVM')

    X, y, vectorizer = prepare_data_for_svm(df_with_topics_and_words)

    print('Train test splitting')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print('---SVM training---')

    svm = train_svm_classifier(X_train, y_train)

    print('---SVM predicting---')
    y_pred = svm.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    return svm, vectorizer, model, df_with_topics_and_words

def train_LDA(df,with_certain_words_removal = False, n_topics = 5, chunksize = 1000, passes = 1, iterations = 100, eval_every = None):
    
    model,texts_bow, dictionary = LDA_training(df, with_certain_words_removal, n_topics, chunksize, passes,iterations, eval_every)
    df_classified = classification(df,model,texts_bow)

    
