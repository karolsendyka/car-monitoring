# normalize class names

# importing the pandas library
import pandas as pd
import os
from pathlib import Path


def csv_to_dirs(csv_name="bugs_to_teams.csv", data_type="train"):
    global index, row, f, description, summary, text
    df = pd.read_csv(csv_name, dtype=str)
    for index, row in df.iterrows():
        row["team"] = row["team"].replace(" ", "")
    df.dropna(inplace=True)
    counter = 0
    for index, row in df.iterrows():
        team = row["team"]
        Path(f'./data/{data_type}/{team}').mkdir(parents=True, exist_ok=True)
        counter += 1
        fileName = f"./data/{data_type}/{team}/{counter}.txt"
        f = open(fileName, "w")

        description = row["Description"]
        summary = row["Summary"]
        text = '\n' + description
        f.write(text)
        f.close()


# reading the csv file
csv_to_dirs("training.csv", "train")
csv_to_dirs("validation.csv", "validation")

