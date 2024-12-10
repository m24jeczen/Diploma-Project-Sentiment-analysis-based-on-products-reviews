import nltk
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases, LdaModel
from gensim.corpora import Dictionary
import pyLDAvis.gensim
import pyLDAvis
from nltk.corpus import stopwords



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

    if with_certain_words_removal:
        product_specific_nouns = set([
            'picture', 'photo', 'jacket', 'dress', 'shoe', 'shirt', 'jeans', 'skirt', 
            'blouse', 'pant', 'hat', 'sock', 'bag', 'watch'  # Expand as needed, idk if it makes any sense
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
    dictionary.filter_extremes(no_below=3, no_above=0.8)

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
    texts_bow, dictionary, id2word = preprocess_text(df)

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


def classification(model):
    print('--Classification starting---')