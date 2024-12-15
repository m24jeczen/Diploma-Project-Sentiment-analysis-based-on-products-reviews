import os
import torch

# Defining directory of the data
current_directory = os.getcwd()
upper_catalog = os.path.abspath(os.path.join(current_directory, ".."))
target_directory = os.path.join(upper_catalog, "amazon_data") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DownloadedModel:
    def __init__(self, model_name, local_path):
        self.model_name = model_name
        self.local_path = local_path

roberta_model = DownloadedModel(model_name = "cardiffnlp/twitter-roberta-base-sentiment", local_path = "../twitter-roberta-sentiment")

bert_model = DownloadedModel(model_name="bert-base-uncased", local_path="../fine_tuned_bert")

categories = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
    "Unknown"
]