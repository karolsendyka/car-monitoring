import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

# Replace with your CSV file path and column name
csv_file_path = 'bugs_to_teams.csv'
text_column = 'Summary'  # Replace with the column containing the text data

# Read CSV into DataFrame
df = pd.read_csv(csv_file_path)

# Drop any missing or NaN values in the text column
texts = df[text_column].dropna().tolist()

# Initialize Keras Tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")  # You can use out-of-vocabulary token if needed

# Fit tokenizer on the texts
tokenizer.fit_on_texts(texts)

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(texts)

# Calculate MAX_SEQUENCE_LENGTH
max_sequence_length = max(len(seq) for seq in sequences)

# Calculate Vocabulary Size
vocabulary_size = len(tokenizer.word_index) + 1  # Adding 1 to account for reserved index 0

# Output results
print(f"MAX_SEQUENCE_LENGTH: {max_sequence_length}")
print(f"Vocabulary Size: {vocabulary_size}")
