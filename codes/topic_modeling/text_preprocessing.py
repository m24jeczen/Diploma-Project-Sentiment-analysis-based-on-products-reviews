from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from nltk.corpus import stopwords
from gensim.models import Phrases
from gensim.corpora import Dictionary
import spacy
nlp = spacy.load("en_core_web_sm")
from sklearn.feature_extraction.text import TfidfVectorizer


def remove_numbers(texts):
    try:
        texts = [[w for w in text if not w.isnumeric()] for text in texts]
        written_numbers = {
                "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                "eighty", "ninety", "hundred", "thousand", "million", "billion"
            }
        texts = [[w for w in text if w not in written_numbers] for text in texts]
        print('--- Numbers and written out numbers removed ---')
        return texts
    except Exception as e:
        print(f"Exception during numbers removal: {e}")
        raise

def remove_verbs(texts):
    try:
        # Process the texts with spaCy in a memory-efficient way
        def filter_verbs(doc):
            # Remove tokens with the part-of-speech tag "VERB"
            return [token.text for token in doc if token.pos_ != 'VERB']

        # Process texts in batches using nlp.pipe for efficiency
        docs = nlp.pipe([' '.join(text) for text in texts], batch_size=1000)
        
        filtered_texts = [filter_verbs(doc) for doc in docs]

        print('--- Verbs removed ---')
        return filtered_texts

    except Exception as e:
        print(f"Exception during removing verbs: {e}")
        raise


def preprocess_text(df, no_below=3, no_above=0.8, with_certain_words_removal=False, 
                    words_to_remove=None, groupby_column='rating'):
    # no_below : Keep tokens which are contained in at least `no_below` documents
    # no_above : Keep tokens which are contained in no more than `no_above` documents
    #        (fraction of total corpus size, not an absolute number)

    try:

        if not pd.api.types.is_string_dtype(df['text']):
            raise TypeError("Column with text has to be string-type.")

        # Handle null values and empty strings in 'text' column
        df['text'] = df['text'].fillna('')  # Replace NaN with empty strings
        if df['text'].str.strip().eq('').any():
            raise ValueError("Column 'text' has only NULL or '' values")

        print('--- Preprocessing starting ---')
        texts = df['text'].astype(str).tolist()

        try:
            # Lower-case and tokenization
            tokenizer = RegexpTokenizer(r'\w+')
            texts = [tokenizer.tokenize(text.lower()) for text in texts]
            print('--- Tokenization done ---')
        except Exception as e:
            print(f"Exception during tokenization: {e}")
            raise

        texts = remove_numbers(texts)

        try:
            # Removing 2-letter words
            texts = [[w for w in text if len(w) > 2] for text in texts]
            print('--- Two letter words removed ---')
        except Exception as e:
            print(f"Exception during removing 2-letter words: {e}")
            raise

        texts = remove_verbs(texts)

        try:
            # Lematization
            lemmatizer = WordNetLemmatizer()
            texts = [[lemmatizer.lemmatize(w, pos='v') for w in text] for text in texts]
            texts = [[lemmatizer.lemmatize(w, pos='n') for w in text] for text in texts]
            print('--- Lematization done ---')
        except Exception as e:
            print(f"Exception during lematization: {e}")
            raise

        try:
            # Removing stopwords
            stop_words = set(stopwords.words('english'))
            texts = [[w for w in text if w not in stop_words] for text in texts]
            print('--- Stopwords removed ---')
        except Exception as e:
            print(f"Exception during removing stopwords: {e}")
            raise

        if with_certain_words_removal:
            try:
                # Removing manually chosen words
                texts = [[w for w in text if w not in words_to_remove] for text in texts]
                print('---Choosed words removed---')
            except Exception as e:
                print(f"Exception during removing chosen words: {e}")
                raise

        try:
            # Bigrams
            bigram = Phrases(texts, min_count=30)
            texts = [[w for w in text] + [w for w in bigram[text] if '_' in w] for text in texts]
            print('--- Bigrams done ---')
        except Exception as e:
            print(f"Exception during finding bigrams: {e}")
            raise

        try:
        # Apply TF-IDF for global word filtering
            all_text_data = [' '.join(text) for text in texts]
            global_tfidf = TfidfVectorizer(stop_words='english', max_features=500)
            global_tfidf_matrix = global_tfidf.fit_transform(all_text_data)
            feature_names = global_tfidf.get_feature_names_out()
            global_word_scores = global_tfidf_matrix.sum(axis=0).A1
            common_words = {word for word, score in zip(feature_names, global_word_scores) if score > 0}
            texts = [[word for word in text if word not in common_words] for text in texts]
            print('--- tfidf global done ---')

        except Exception as e:
            print(f"Exception during global word filtering: {e}")    
        try:
            dictionary = Dictionary(texts)
            dictionary.filter_extremes(no_below, no_above)
            print('--- Common and rare words removed ---')
        except Exception as e:
            print(f"Exception during removing common and rare words: {e}")
 
        try:
            if groupby_column:
                grouped_texts = []
                for _, group in df.groupby(groupby_column):
                    group_texts = group['text'].astype(str).tolist()
                    group_texts = [' '.join(tokenizer.tokenize(text.lower())) for text in group_texts]
                    grouped_texts.extend(group_texts)

                global_tfidf_matrix = global_tfidf.fit_transform(grouped_texts)
                print('--- tfidf grouped done ---')

        except Exception as e:
            print(f"Exception during groupby column: {e}")
        
        try:
            # Creating bag of words
            texts_bow = [dictionary.doc2bow(text) for text in texts]
            id2token = {id: token for token, id in dictionary.token2id.items()}
            print('--- Preprocessing done ---')
            return texts_bow, dictionary, id2token
        except Exception as e:
            print(f"Exception during making bag-of-words: {e}")
            raise

    except Exception as final_error:
        print(f"Unexpected exception in preprocessing text: {final_error}")
        raise