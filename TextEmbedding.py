import os
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from openai.embeddings_utils import cosine_similarity

# Set your OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "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"

# Define a dictionary containing words you want to embed
data = {"Words": ["College", "Car", "Student", "Orange"]}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Initialize the OpenAIEmbeddings object
embeddings = OpenAIEmbeddings()

# Create an empty list to store word embeddings
embeddings_list = []

# Iterate through each word in the DataFrame and calculate its embedding
for index, row in df.iterrows():
    word = row["Words"]
    embedding = embeddings.embed_query(word)
    embeddings_list.append(embedding)

# Add the embeddings to the DataFrame
df["embeddings"] = embeddings_list

# Define a text for which you want to calculate similarity scores
our_text = "Apple"

# Calculate the embedding for the given text
text_embedding = embeddings.embed_query(our_text)

# Create an empty list to store similarity scores
similarity_score_list = []

# Iterate through each row in the DataFrame and calculate similarity scores
for i, r in df.iterrows():
    embeddings = r["embeddings"]
    similarity_score = cosine_similarity(embeddings, text_embedding)

    # Multiply the similarity score by 100 to express it as a percentage
    similarity_score_list.append(similarity_score * 100)

# Add the similarity scores to the DataFrame
df["similarity score"] = similarity_score_list

# Sort the DataFrame based on similarity scores in descending order
df = df.sort_values("similarity score", ascending=False)

# Print the final DataFrame
print(df)
