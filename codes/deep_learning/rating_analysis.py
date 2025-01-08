import pandas as pd
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
