import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def topic_distributions_to_matrix(model, texts_bow, n_topics):
    topic_distributions = [model.get_document_topics(bow, minimum_probability=0.0) for bow in texts_bow]
    feature_matrix = []
    for dist in topic_distributions:
        topic_probs = [0.0] * n_topics
        for topic_id, prob in dist:
            topic_probs[topic_id] = prob
        feature_matrix.append(topic_probs)
    return feature_matrix


def add_sentiment_column(df):
    df['sentiment'] = df['rating'].apply(lambda x: 'negative' if x <=2 else ('neutral' if x==3 else 'positive'))
    df['sentiment_encoded'] = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
    return df

def add_sentiment_column2(df):
    df['sentiment'] = df['rating'].apply(lambda x: 'negative' if x <=3 else 'positive')
    return df

def train_test_spliter(topic_dist_matrix, labels):
    X_train, X_test, y_train, y_test = train_test_split(topic_dist_matrix,labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_and_evaluate_Random_Forest(topic_dist_matrix, labels,
                                      X_train=None, X_test=None, y_train=None, y_test=None):
    ret_datasets = False
    if X_train is None or X_test is None or y_train is None or y_test is None:
        ret_datasets = True
        X_train, X_test, y_train, y_test = train_test_split(topic_dist_matrix,labels, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)

    print('---Random Forest Training ---')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy for Random Forest: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Classification Report for Random Forest:")
    print(classification_report(y_test, y_pred))
    print("\n")

    if ret_datasets:
        return model,X_train, X_test, y_train, y_test, y_pred
    else:
        return model, y_pred


def train_and_evaluate_Logistic_Regression(topic_dist_matrix, labels,
                                            X_train=None, X_test=None, y_train=None, y_test=None):
    ret_datasets = False
    if X_train is None or X_test is None or y_train is None or y_test is None:
        ret_datasets = True
        X_train, X_test, y_train, y_test = train_test_split(topic_dist_matrix,labels, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=200, random_state=42)

    print('---Logistic Regression Training ---')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy for Logistic Regression: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Classification Report for Logistic Regression:")
    print(classification_report(y_test, y_pred))
    print("\n")

    if ret_datasets:
        return model,X_train, X_test, y_train, y_test, y_pred
    else:
        return model, y_pred

def train_and_evaluate_Gradient_Boosting(topic_dist_matrix, labels,
                                          X_train=None, X_test=None, y_train=None, y_test=None):
    ret_datasets = False
    if X_train is None or X_test is None or y_train is None or y_test is None:
        ret_datasets = True
        X_train, X_test, y_train, y_test = train_test_split(topic_dist_matrix,labels, test_size=0.2, random_state=42)
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42) 

    print('--- Gradient Boosting Training ---')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy for Gradient Boosting: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Classification Report for Gradient Boosting:")
    print(classification_report(y_test, y_pred))
    print("\n")

    if ret_datasets:
        return model,X_train, X_test, y_train, y_test, y_pred
    else:
        return model, y_pred



def train_and_evaluate_classifier(topic_dist_matrix, labels, 
                                  X_train=None, X_test=None, y_train=None, y_test=None):
    if X_train is None or X_test is None or y_train is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(topic_dist_matrix,labels, test_size=0.2, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42) 
    }

    for model_name, model in models.items():
        print(f'--- Training of {model_name} ---')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"Accuracy for {model_name}: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred))
        print("\n")

    
def compute_confusion_matrix(y_test, y_pred, num_classes):
    conf_matrix = confusion_matrix(y_test, y_pred, labels=range(num_classes))

    metrics = {cls: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for cls in range(num_classes)}

    # Compute TP, FP, FN, TN for each class
    for cls in range(num_classes):
        TP = conf_matrix[cls, cls]  # True Positives for the class
        FP = conf_matrix[:, cls].sum() - TP  # False Positives for the class
        FN = conf_matrix[cls, :].sum() - TP  # False Negatives for the class
        TN = conf_matrix.sum() - (TP + FP + FN)  # True Negatives for the class
        
        metrics[cls]["TP"] = TP
        metrics[cls]["FP"] = FP
        metrics[cls]["FN"] = FN
        metrics[cls]["TN"] = TN

    return metrics

def create_cumulative_matrix(y_test, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    if num_classes==3:
        labels = ['negative', 'neutral', 'positive']
    else:
        labels = ['negative', 'positive']


    for true, pred in zip(y_test, y_pred):
        matrix[true, pred] += 1  # Increment the corresponding cell

    #matrix_df = pd.DataFrame(matrix, index=[f"Real {i}" for i in range(num_classes)], 
    #                         columns=[f"Pred {i}" for i in range(num_classes)])
    
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
    plt.title("Cumulative Matrix Heatmap")
    plt.xlabel("Predicted Values")
    plt.ylabel("Real Values")
    plt.show()


    return matrix_df

