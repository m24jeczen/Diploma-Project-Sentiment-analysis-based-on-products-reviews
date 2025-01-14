import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def heatmap(df,column1,column2):

    cross_tab = pd.crosstab(df[column1], df[column2])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cross_tab, annot=True, cmap="coolwarm", fmt="d")
    plt.title("Heatmap of correlation beetwen true values and prediction")
    plt.xlabel(column2)
    plt.ylabel(column1)
    plt.show()

def average_rating_per_month(df):
    df
    plt.figure(figsize=(10, 6))
    plt.plot(df["miesiac"], df["score"], marker="o", linestyle="-", color="b", label="Averaged score thru months")
    plt.title("Averaged score thru months")
    plt.xlabel("Time")
    plt.ylabel("Averaged Score")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_number_of_words_percentages(data, limit = 500):
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