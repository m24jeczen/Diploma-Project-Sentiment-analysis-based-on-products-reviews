import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from io import BytesIO


def calculate_metrics(df, label_col='label', prediction_col='prediction'):
    if label_col not in df.columns or prediction_col not in df.columns:
        raise ValueError(f"Tabel must contains columns '{label_col}' and '{prediction_col}'.")

    mae = mean_absolute_error(df[label_col], df[prediction_col])
    avg_accuracy = accuracy_score(df[label_col], df[prediction_col])

    unique_labels = df[label_col].unique()
    metrics_per_label = []

    for label in unique_labels:
        true_positive = ((df[label_col] == label) & (df[prediction_col] == label)).sum()
        total_for_label = (df[label_col] == label).sum()
        label_accuracy = true_positive / total_for_label 
        label_f1 = f1_score(df[label_col], df[prediction_col], labels=[label], average='macro')
        mae_for_label = mean_absolute_error(
            df[df[label_col] == label][label_col],
            df[df[label_col] == label][prediction_col]
        )

        metrics_per_label.append({
            'label': label,
            'accuracy': label_accuracy,
            'f1_score': label_f1,
            'mae': mae_for_label
        })

    metrics_df = pd.DataFrame(metrics_per_label)

    return {
        'MAE': mae,
        'Average Accuracy': avg_accuracy,
        'Metrics Per Label': metrics_df
    }


def heatmap(df, column1, column2):
    # Creating a cross-tabulation for heatmap
    cross_tab = pd.crosstab(df[column1], df[column2])

    # Creating the heatmap with Matplotlib and Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cross_tab, annot=True, cmap="coolwarm", fmt="d")
    plt.title("Heatmap of Correlation Between True Values and Predictions")
    plt.xlabel(column2)
    plt.ylabel(column1)

    # Save the heatmap to a BytesIO stream for Streamlit compatibility
    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", bbox_inches="tight")
    img_stream.seek(0)  # Reset the stream to the beginning for reading
    plt.close()

    # Return the image stream for Streamlit
    return img_stream


def plot_number_of_words_percentages(data, limit=500):
    numbers = data["text"].str.split().str.len()
    sorted_numbers = np.sort(numbers)
    cumulative_probabilities = np.arange(1, len(sorted_numbers) + 1) / len(sorted_numbers)


    plt.figure(figsize=(10, 6))
    plt.plot(sorted_numbers, cumulative_probabilities)

    plt.title('Cumulative Distribution of Words Length of Reviews', fontsize=16)
    plt.xlabel('Words per review', fontsize=14)
    plt.ylabel('Percentage of all reviews', fontsize=14)

    plt.grid(alpha=0.7)
    plt.xlim(0, limit)
    plt.ylim(bottom=0)
    plt.xticks(np.arange(0, limit, limit//10))

# Ustawianie znaczników na osi Y co 0.1
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()


def plot_monthly_avg(data, label = "rating",pointers = 20):
    data['date'] = pd.to_datetime(data['timestamp'])
    # Usuń wiersze z nieprawidłowymi datami
    data = data.dropna(subset=['date'])

    # Dodanie kolumny z miesiącem i rokiem w formacie 'MM-YYYY'
    data['date'] = data['date'].dt.strftime('%Y-%m-%d')

    # Posortuj ramkę danych po dacie
    data = data.sort_values(by='date')

    # Oblicz skumulowaną średnią
    data['cumulative_average'] = data[label].expanding().mean()
    markers =  np.linspace(0, len(data) - 1, pointers, dtype=int)
    plot_data = data.iloc[markers]

    plt.figure(figsize=(10, 6))
    plt.plot(plot_data['date'], plot_data['cumulative_average'], marker='o')
    plt.title('Change of average score through time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average score', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def distribiution_of_rating(df, label = "rating"):
    rating_counts = df[label].value_counts(normalize=True) * 100  # Normalize=True gives proportions
    rating_counts = rating_counts.sort_index()  # Ensure ratings are sorted from 1 to 5

    # Bar positions and width
    ratings = rating_counts.index
    percentages = rating_counts.values
    bar_width = 0.6  # Width of the bars

    # Create the bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(ratings, percentages, width=bar_width, edgecolor='black', align='center')

    # Add percentage values above the bars
    for i, percentage in enumerate(percentages):
        plt.text(ratings[i], percentage + 1, f'{percentage:.1f}%', ha='center', fontsize=10)

    # Add labels and title
    plt.title('Percentage Frequency of Ratings', fontsize=16)
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Percentage Frequency (%)', fontsize=12)
    plt.xticks(ticks=ratings, labels=ratings)  # Ensure x-ticks are aligned with ratings
    plt.ylim(0, max(percentages) + 5)  # Add some padding above the tallest bar

    # Ensure no bars are touching
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()