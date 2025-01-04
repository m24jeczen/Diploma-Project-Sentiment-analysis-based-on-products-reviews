import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def heatmap(df,column1,column2):

    # Tworzenie macierzy współwystępowania (np. liczba par A-B, A-C itp.)
    cross_tab = pd.crosstab(df[column1], df[column2])

    # Tworzenie heatmapy
    plt.figure(figsize=(8, 6))
    sns.heatmap(cross_tab, annot=True, cmap="coolwarm", fmt="d")
    plt.title("Heatmap of correlation beetwen true values and prediction")
    plt.xlabel(column2)
    plt.ylabel(column1)
    plt.show()
    