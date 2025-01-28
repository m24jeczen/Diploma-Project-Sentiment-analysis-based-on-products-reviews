import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from io import BytesIO
import matplotlib
import re 
import matplotlib.pyplot as plt


def calculate_metrics(df, label_col='label', prediction_col='prediction'):
    if label_col not in df.columns or prediction_col not in df.columns:
        raise ValueError(f"Table must contain columns '{label_col}' and '{prediction_col}'.")

    mae = round(mean_absolute_error(df[label_col], df[prediction_col]), 5)
    avg_accuracy = round(accuracy_score(df[label_col], df[prediction_col]), 5)

    unique_labels = df[label_col].unique()
    metrics_per_label = []

    for label in unique_labels:
        true_positive = ((df[label_col] == label) & (df[prediction_col] == label)).sum()
        total_for_label = (df[label_col] == label).sum()
        label_accuracy = round(true_positive / total_for_label, 5) if total_for_label != 0 else 0
        label_f1 = round(f1_score(df[label_col], df[prediction_col], labels=[label], average='macro'), 5)
        mae_for_label = round(mean_absolute_error(
            df[df[label_col] == label][label_col],
            df[df[label_col] == label][prediction_col]
        ), 5)

        metrics_per_label.append({
            'Label': label,
            'Accuracy': label_accuracy,
            'F-score': label_f1,
            'MAE': mae_for_label
        })

    # Convert metrics to DataFrame and round values
    metrics_df = pd.DataFrame(metrics_per_label).round(3).sort_values(by='Label')
    metrics_df = metrics_df.reset_index(drop=True)
    return {
        'MAE': mae,
        'Average Accuracy': avg_accuracy,
        'Metrics Per Label': metrics_df
    }



def heatmap(df, column1, column2):

    fixed1 = column1.replace("_", " ").capitalize()
    fixed2 = column2.replace("_", " ").capitalize()

    # Creating a cross-tabulation for the heatmap
    cross_tab = pd.crosstab(df[column1], df[column2])

    # App theme colors
    primary_color = "#F6B17A"
    secondary_background_color = "#424769"
    text_color = "#E2E8F0"
    background_color = "#1C2138"

    # Custom colormap: Shades from secondary background to primary color
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_theme", [secondary_background_color, primary_color], N=256
    )

    # Creating the heatmap with Matplotlib and Seaborn
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cross_tab,
        annot=True,
        cmap=custom_cmap,
        fmt="d",
        cbar_kws={'shrink': 0.8, 'format': '%.0f'},
        annot_kws={"color": text_color}
    )

    # Customize the color bar font color
    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(colorbar.ax.yaxis.get_majorticklabels(), color=text_color)

    # Customizing the heatmap's appearance to match the app theme
    plt.title("Confusion Matrix of True Values and Predictions", color=text_color)
    plt.xlabel(fixed2, color=text_color)
    plt.ylabel(fixed1, color=text_color)

    # Set axis label colors
    ax.tick_params(colors=text_color)

    # Change the background color of the figure
    ax.figure.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", bbox_inches="tight", facecolor=background_color)
    img_stream.seek(0)  
    plt.close()

    return img_stream

def plot_number_of_words_percentages_for_app(data, limit=500):
    # Update the color theme from the app configuration
    primary_color = "#F6B17A"  # Line color
    background_color = "#1C2138"  # Entire image background
    secondary_background_color = "#424769"  # Gridlines and plot background
    text_color = "#E2E8F0"  # Text color
    light_grid_color = "#5B637C"  # Lighter gridlines

    # Data processing
    numbers = data["text"].str.split().str.len()
    sorted_numbers = np.sort(numbers)
    cumulative_probabilities = 100 * np.arange(1, len(sorted_numbers) + 1) / len(sorted_numbers)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_numbers, cumulative_probabilities, color=primary_color)
    plt.title('Cumulative Distribution of Words Length of Reviews', fontsize=14, color=text_color)
    plt.xlabel('Words per review', fontsize=12, color=text_color)
    plt.ylabel('Percentage of all reviews (%)', fontsize=12, color=text_color)
    plt.xticks(np.arange(0, limit, limit // 10), color=text_color)
    plt.yticks(np.arange(0, 101, 10), color=text_color)  # Adjusted to percentages (0-100)
    plt.xlim(0, limit)
    plt.ylim(0, 100)  # Set percentage range

    # Explicitly set tick colors for both axes
    plt.gca().tick_params(axis='x', colors=text_color)
    plt.gca().tick_params(axis='y', colors=text_color)

    plt.grid(True, color=light_grid_color, linestyle='--', linewidth=0.5)
    plt.gca().set_facecolor(secondary_background_color)
    plt.tight_layout()

    # Set the overall background color
    plt.gcf().patch.set_facecolor(background_color)

    # Save plot to a BytesIO stream
    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", bbox_inches="tight", facecolor=background_color)
    img_stream.seek(0)
    plt.close()
    return img_stream



def plot_monthly_avg_app(data, label="rating", pointers=20):
    primary_color = "#F6B17A"  
    background_color = "#1C2138"  
    secondary_background_color = "#424769"  
    text_color = "#E2E8F0"  
    light_grid_color = "#5B637C"  

    data['date'] = pd.to_datetime(data['timestamp'])
    data = data.dropna(subset=['date'])
    data['date'] = data['date'].dt.strftime('%Y-%m-%d')
    data = data.sort_values(by='date')
    data['cumulative_average'] = data[label].expanding().mean()
    markers = np.linspace(0, len(data) - 1, pointers, dtype=int)
    plot_data = data.iloc[markers]

    plt.figure(figsize=(10, 6))
    plt.plot(plot_data['date'], plot_data['cumulative_average'], marker='o', color=primary_color)
    plt.title(f'Change of {label} score through time', fontsize=14, color=text_color)
    plt.xlabel('Date', fontsize=12, color=text_color)
    plt.ylabel(f'Average {label}', fontsize=12, color=text_color)
    plt.xticks(rotation=45, color=text_color)
    plt.yticks(color=text_color)

    plt.gca().tick_params(axis='x', colors=text_color)
    plt.gca().tick_params(axis='y', colors=text_color)

    plt.grid(True, color=light_grid_color, linestyle='--', linewidth=0.5)
    plt.gca().set_facecolor(secondary_background_color)
    plt.tight_layout()

    plt.gcf().patch.set_facecolor(background_color)

    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", bbox_inches="tight", facecolor=background_color)
    img_stream.seek(0)
    plt.close()
    return img_stream


def distribiution_of_rating_for_app(df, label="rating"):
    primary_color = "#F6B17A"  
    background_color = "#1C2138"  
    secondary_background_color = "#424769"  
    text_color = "#E2E8F0"  

    matplotlib.rcParams.update({
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'axes.edgecolor': text_color,
        'figure.facecolor': background_color,
        'axes.facecolor': background_color
    })

    rating_counts = df[label].value_counts(normalize=True) * 100  
    rating_counts = rating_counts.sort_index()  

    ratings = rating_counts.index
    percentages = rating_counts.values
    bar_width = 0.6  

    plt.figure(figsize=(8, 6))
    plt.bar(ratings, percentages, width=bar_width, edgecolor='black', align='center', color=primary_color)

    for i, percentage in enumerate(percentages):
        plt.text(ratings[i], percentage + 1, f'{percentage:.1f}%', ha='center', fontsize=10, color=text_color)

    plt.title(f'Percentage Frequency of {label}', fontsize=16, color=text_color)
    plt.xlabel(label, fontsize=12)
    plt.ylabel('Percentage Frequency (%)', fontsize=12)
    plt.xticks(ticks=ratings, labels=ratings)  
    plt.ylim(0, max(percentages) + 5)  

    plt.grid(axis='y', linestyle='--', alpha=0.7, color=text_color)
    plt.tight_layout()

    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", bbox_inches="tight")
    img_stream.seek(0)
    plt.close()
    return img_stream
