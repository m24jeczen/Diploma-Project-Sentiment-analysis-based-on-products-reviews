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
import gensim


def preprocess_text(df, with_certain_words_removal=False, data_is_big=False):
    try:
        # Sprawdzenie wstępne
        # Preliminary check for 'text' column and data types
        if 'text' not in df.columns:
            raise KeyError("Column 'text' does not exist in data.")
        
        # Ensure 'text' column is of string type
        if not pd.api.types.is_string_dtype(df['text']):
            raise TypeError("Column 'text' has to be string-type.")
        
        df['text'] = df['text'].fillna('')  # Replace NaN with empty strings
        if df['text'].str.strip().eq('').any():
            raise ValueError("Column 'text' has only NULL or '' values")

        # Handle null values and empty strings in 'text' column
        df['text'] = df['text'].fillna('')  # Replace NaN with empty strings
        if df['text'].str.strip().eq('').any():
            raise ValueError("Column 'text' has only NULL or '' values")

        texts = df['text'].astype(str).tolist()
        print('Tokenization starting ---')

        try:
            # Lower-case i tokenizacja
            tokenizer = RegexpTokenizer(r'\w+')
            texts = [tokenizer.tokenize(text.lower()) for text in texts]
            print('Tokenization done')
        except Exception as e:
            print(f"Exception during tokenization: {e}")
            raise

        try:
            # Usuwanie liczb
            texts = [[w for w in text if not w.isnumeric()] for text in texts]
            print('Numbers removed')
        except Exception as e:
            print(f"Exception during numbers removal: {e}")
            raise

        try:
            # Usuwanie 2-literowych słów
            texts = [[w for w in text if len(w) > 2] for text in texts]
            print('Two letter words removed')
        except Exception as e:
            print(f"Exception during removing 2-letter words: {e}")
            raise

        try:
            # Usuwanie zapisanych słownie liczb
            written_numbers = {
                "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                "eighty", "ninety", "hundred", "thousand", "million", "billion"
            }
            texts = [[w for w in text if w not in written_numbers] for text in texts]
            print('Written-out numbers removed')
        except Exception as e:
            print(f"Exception during removing written-out numbers: {e}")
            raise

        if data_is_big:
            try:
                # Usuwanie czasowników (duże dane)
                common_verbs = set([
                    'be', 'have', 'do', 'say', 'make', 'go', 'know', 'take', 'see', 'get',
                    'give', 'come', 'think', 'look', 'want', 'use', 'find', 'tell', 'ask',
                    'work', 'seem', 'feel', 'try', 'leave', 'call', 'need', 'become', 'put',
                    'mean', 'keep', 'let', 'begin', 'seem', 'help', 'talk', 'turn', 'start',
                    'show', 'hear', 'play', 'run', 'move', 'live', 'believe', 'hold', 'bring', 'wear', 'come', 'use', 'work'
                ])
                texts = [[word for word in text if word not in common_verbs] for text in texts]
                print('Verbs removed BIG')
            except Exception as e:
                print(f"Exception during removing verbs: {e}")
                raise
        else:
            try:
                # Usuwanie czasowników (małe dane)
                texts = [
                    [word for word, tag in pos_tag(word_tokenize(' '.join(text))) if tag[:2] != 'VB']
                    for text in texts
                ]
                print('Verbs removed SMALL')
            except Exception as e:
                print(f"Exception during removing verbs: {e}")
                raise

        try:
            # Lematization
            lemmatizer = WordNetLemmatizer()
            texts = [[lemmatizer.lemmatize(w, pos='v') for w in text] for text in texts]
            texts = [[lemmatizer.lemmatize(w, pos='n') for w in text] for text in texts]
            print('Lematization done')
        except Exception as e:
            print(f"Exception during lematization: {e}")
            raise

        try:
            # Usuwanie stopwords
            stop_words = set(stopwords.words('english'))
            texts = [[w for w in text if w not in stop_words] for text in texts]
            print('Stopwords removed')
        except Exception as e:
            print(f"Exception during removing stopwords: {e}")
            raise

        if with_certain_words_removal:
            try:
                # Usuwanie określonych słów
                product_specific_nouns = set([
                    'picture', 'photo', 'jacket', 'dress', 'shoe', 'shirt', 'jeans', 'skirt',
                    'blouse', 'pant', 'hat', 'sock', 'bag', 'watch', 'get', 'wear', 'like', 'watch'
                ])
                texts = [[w for w in text if w not in product_specific_nouns] for text in texts]
                print('Choosed words removed')
            except Exception as e:
                print(f"Exception during removing chosen words: {e}")
                raise

        try:
            # Bigramy
            bigram = Phrases(texts, min_count=30)
            texts = [[w for w in text] + [w for w in bigram[text] if '_' in w] for text in texts]
            print('Bigrams done')
        except Exception as e:
            print(f"Exception during finding bigrams: {e}")
            raise

        try:
            # Tworzenie słownika
            dictionary = Dictionary(texts)
            # Przekształcanie do bag-of-words
            texts_bow = [dictionary.doc2bow(text) for text in texts]
            id2token = {id: token for token, id in dictionary.token2id.items()}
            print('Preprocessing done')
            return texts_bow, dictionary, id2token
        except Exception as e:
            print(f"Exception during making bad-of-words: {e}")
            raise

    except Exception as final_error:
        print(f"Unexpected exception in preprocessing text: {final_error}")
        raise



def display_visuals_LDA(model, texts_bow, dictionary):
    try:
        # Check if the inputs are valid
        if model is None or not texts_bow or dictionary is None:
            raise ValueError("Invalid inputs provided to display_visuals_LDA. Check the model, texts_bow, and dictionary.")

        # Prepare the LDA visualization
        LDAvis_prepared = pyLDAvis.gensim.prepare(model, texts_bow, dictionary)
        return LDAvis_prepared
    except ValueError as ve:
        print(f"ValueError in display_visuals_LDA: {ve}")
    except ImportError:
        print("pyLDAvis library is not installed or improperly imported. Please install it using `pip install pyLDAvis`.")
    except Exception as e:
        print(f"An unexpected error occurred in display_visuals_LDA: {e}")
    return None


def display_top_words_for_topics(lda_model, n_topics, n_words=10):
    try:
        # Check if the model and number of topics are valid
        if lda_model is None or n_topics <= 0:
            raise ValueError("Invalid LDA model or number of topics provided.")

        for topic_id in range(n_topics):
            try:
                # Get the top words for the topic
                top_words = lda_model.show_topic(topic_id, topn=n_words)
                words = [word for word, _ in top_words]
                print(f"Topic {topic_id}: {', '.join(words)}")
            except Exception as e:
                print(f"Error retrieving words for topic {topic_id}: {e}")
    except ValueError as ve:
        print(f"ValueError in display_top_words_for_topics: {ve}")
    except AttributeError:
        print("The provided LDA model does not have the required method `show_topic`. Check the model type.")
    except Exception as e:
        print(f"An unexpected error occurred in display_top_words_for_topics: {e}")




from gensim.models import LdaModel

def LDA_training(df, with_certain_words_removal=False, n_topics=5, chunksize=1000, passes=100, iterations=200, update_every=1, eval_every=float('inf'), texts_bow=None, dictionary=None, id2word=None):
    
    # Check if 'text' column exists in the dataframe
    if 'text' not in df.columns:
        raise KeyError("Column 'text' does not exist in data.")
    
    # Validate types of arguments
    if not isinstance(with_certain_words_removal, bool):
        raise TypeError("Argument 'with_certain_words_removal' must be a boolean.")
    
    if not isinstance(n_topics, int) or n_topics <= 0:
        raise ValueError("Argument 'n_topics' must be a positive integer.")
    
    if not isinstance(chunksize, int) or chunksize <= 0:
        raise ValueError("Argument 'chunksize' must be a positive integer.")
    
    if not isinstance(passes, int) or passes <= 0:
        raise ValueError("Argument 'passes' must be a positive integer.")
    
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("Argument 'iterations' must be a positive integer.")
    
    if not isinstance(update_every, int) or update_every <= 0:
        raise ValueError("Argument 'update_every' must be a positive integer.")
    
    if not isinstance(eval_every, (int, float)):
        raise TypeError("Argument 'eval_every' must be an integer or a float.")
    
    if texts_bow is None or dictionary is None or id2word is None:
        # If necessary inputs are missing, preprocess the text data
        if 'text' not in df.columns:
            raise KeyError("DataFrame must contain a 'text' column for preprocessing.")
        
        # Assuming preprocess_text is a predefined function
        texts_bow, dictionary, id2word = preprocess_text(df, with_certain_words_removal=with_certain_words_removal)

    # Ensure that texts_bow is in the correct format (list of lists of tuples)
    if not isinstance(texts_bow, list) or not all(isinstance(doc, list) and all(isinstance(word, tuple) and len(word) == 2 for word in doc) for doc in texts_bow):
        raise TypeError("Argument 'texts_bow' must be a list of lists of word-frequency tuples.")
    
    # Ensure that dictionary is a gensim Dictionary
    if not isinstance(dictionary, gensim.corpora.Dictionary):
        raise TypeError("Argument 'dictionary' must be a gensim Dictionary object.")
    
    # Ensure that id2word is the same as dictionary
    if not isinstance(id2word, dict):
        raise TypeError("Argument 'id2word' must be a dictionary mapping word IDs to words.")
    
    # Check that the number of topics is consistent with the data
    if n_topics > len(dictionary):
        raise ValueError("Number of topics must not exceed the number of unique words in the dictionary.")

    print('--- Model starting ---')
    
    # LDA model training
    model = LdaModel(corpus=texts_bow, id2word=id2word, chunksize=chunksize, 
                     alpha='auto', eta='auto', 
                     iterations=iterations, num_topics=n_topics, 
                     passes=passes, update_every=update_every, eval_every=eval_every)   

    # Evaluating the model
    top_topics = model.top_topics(texts_bow, topn=3)

    # Calculate and display average topic coherence
    avg_topic_coherence = sum([t[1] for t in top_topics]) / n_topics
    print('Medium koherence of topics: %.4f.' % avg_topic_coherence)

    # Display top words for each topic
    display_top_words_for_topics(model, n_topics=n_topics)

    return model, texts_bow, dictionary




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



