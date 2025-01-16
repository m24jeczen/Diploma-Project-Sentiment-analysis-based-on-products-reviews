from gensim.models import LdaModel
import pyLDAvis
import gensim


def LDA_training(df, texts_bow, dictionary, id2word,
                 n_topics=5, chunksize=1000, passes=100, iterations=200, 
                 update_every=1, eval_every=100):
    
    validate_types_of_arguments(df, texts_bow, dictionary, id2word,
                 n_topics=5, chunksize=1000, passes=100, iterations=200, 
                 update_every=1, eval_every=100)
    

    print('--- LDA starting ---')
    
    # LDA model training
    model = LdaModel(corpus=texts_bow, id2word=id2word, chunksize=chunksize, 
                     alpha='auto', eta='auto', 
                     iterations=iterations, num_topics=n_topics, 
                     passes=passes, update_every=update_every, eval_every=eval_every)   
    print('--- LDA finished ---')

    # Evaluating the model
    top_topics = model.top_topics(texts_bow, topn=20)
    unique_topic_words = make_words_unique_per_topic(model, n_topics=n_topics, topn=20)

    # Calculate and display average topic coherence
    avg_topic_coherence = sum([t[1] for t in top_topics]) / n_topics
    print('Medium koherence of topics: %.4f.' % avg_topic_coherence)

    # Display top words for each topic
    #display_top_words_for_topics(model, n_topics=n_topics)

    for topic_id, words in unique_topic_words.items():
        print(f"Topic {topic_id}: {', '.join(words)}")
    return model

from collections import defaultdict

def make_words_unique_per_topic(model, n_topics, topn=20):
    # Extract top words for each topic
    topic_words = {}
    word_weights = defaultdict(list)

    for topic_id in range(n_topics):
        top_words = model.show_topic(topic_id, topn=topn)
        topic_words[topic_id] = {word: weight for word, weight in top_words}
        for word, weight in top_words:
            word_weights[word].append((topic_id, weight))

    # Resolve overlaps
    unique_topic_words = {topic_id: set() for topic_id in range(n_topics)}
    for word, occurrences in word_weights.items():
        if len(occurrences) == 1:
            # If the word appears in only one topic, keep it there
            topic_id, _ = occurrences[0]
            unique_topic_words[topic_id].add(word)
        else:
            # If the word appears in multiple topics, assign it to the one with the highest weight
            occurrences.sort(key=lambda x: x[1], reverse=True)
            best_topic_id = occurrences[0][0]
            unique_topic_words[best_topic_id].add(word)

    # Replace topic words with unique words
    for topic_id in range(n_topics):
        topic_words[topic_id] = {word: topic_words[topic_id][word]
                                 for word in unique_topic_words[topic_id]}
    
    return topic_words

# After LDA model training




def display_top_words_for_topics(lda_model, n_topics, n_words=20):
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


def validate_types_of_arguments(df, texts_bow, dictionary, id2word,
                 n_topics=5, chunksize=1000, passes=100, iterations=200, 
                 update_every=1, eval_every=float('inf')):
    if 'text' not in df.columns:
        raise KeyError("Column 'text' does not exist in data.")
    
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
    
    else:
        print('--- Arguments are correct ---')



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







