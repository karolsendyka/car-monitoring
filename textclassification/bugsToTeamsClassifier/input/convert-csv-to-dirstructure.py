import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load the CSV file into a pandas DataFrame
def split_csv(input_file):
    # Read the input CSV file
    data = pd.read_csv(input_file)
    print("Number of examples per class:")
    print(data['team'].value_counts())

    # Split the data into 90% training and 10% test
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42, shuffle=True)

    # Save the training data to "training.csv"
    train_data.to_csv('./20241014/training.csv', index=False)
    print(len(train_data))
    # Save the test data to "test.csv"
    test_data.to_csv('./20241014/test.csv', index=False)
    print(len(test_data))

def removeFile(filename):
    try:
        os.remove(filename)
    except OSError:
        pass




def csv_to_dirs(csv_name="bugsToTeams_20241014.csv", data_type="train"):
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
        text = summary
        f.write(text)
        f.close()


removeFile('./20241014/training.csv')
removeFile('./20241014/test.csv')
removeFile('./data/train')
removeFile('./data/validation')
split_csv('./20241014/bugsToTeams_20241014.csv')

csv_to_dirs("20241014/training.csv", "train")
csv_to_dirs("20241014/test.csv", "validation")

