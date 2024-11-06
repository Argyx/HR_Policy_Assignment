# Import necessary libraries
import os
import glob
import re  # For regex-based text splitting
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from kneed import KneeLocator
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import html  # For escaping HTML content
from sklearn.metrics.pairwise import cosine_similarity
import nltk  # For stopwords
from nltk.corpus import stopwords

# Download necessary NLTK data if not already downloaded
nltk.download('stopwords')

# ------------------------ Configuration ------------------------

# Define the path to the folder containing the .txt files
folder_path = 'data'

# Define model parameters
max_tokens = 128  # Maximum number of tokens per chunk, aligned with model's max_seq_length

# Initialize the tokenizer associated with the pre-trained model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Initialize the SentenceTransformer model for generating embeddings
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Qdrant Configuration for vector database
qdrant_host = 'localhost'  # Qdrant host
qdrant_port = 6333         # Qdrant port
qdrant_collection_name = 'text_clusters'  # Name of the collection in Qdrant

# Get Greek stopwords and enhance the list
greek_stopwords = set(stopwords.words('greek'))
# Add additional common words to the stopwords set
additional_stopwords = {'ένα', 'πριν', 'από', 'προς', 'τους', 'στην', 'στις', 'της', 'του', 'μεταξύ', 'ή', 'και'}
greek_stopwords.update(additional_stopwords)

# ------------------------ Functions ------------------------


def split_text_into_chunks(text, max_tokens=128):
    """
    Splits text into chunks based on sentences using regex, ensuring:
    - Each chunk starts and ends with complete sentences.
    - Chunks do not exceed max_tokens.
    - Handles leading and trailing punctuation and whitespace.

    Parameters:
    - text (str): The input text to split.
    - max_tokens (int): Maximum number of tokens per chunk.

    Returns:
    - chunks (list): A list of text chunks.
    """
    # Split text into sentences using regex that matches sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Initialize an empty list to store the chunks
    chunks = []
    # Initialize variables for the current chunk and its token count
    current_chunk = []
    current_chunk_tokens = 0

    # Initialize a counter for the sentences
    i = 0
    # Loop through all sentences
    while i < len(sentences):
        # Get the current sentence and strip leading/trailing whitespace
        sentence = sentences[i].strip()
        # Tokenize the sentence using the tokenizer
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        # Get the number of tokens in the sentence
        sentence_token_count = len(sentence_tokens)

        # Check if adding this sentence would exceed the max_tokens limit
        if current_chunk_tokens + sentence_token_count <= max_tokens:
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            # Update the token count
            current_chunk_tokens += sentence_token_count
            # Move to the next sentence
            i += 1
        else:
            # If the current chunk is not empty, finalize it
            if current_chunk:
                # Join the sentences in the current chunk into a single string
                chunk_text = ' '.join(current_chunk).strip()
                # Remove leading/trailing periods and whitespace
                chunk_text = chunk_text.strip('.').strip()
                # Add the chunk to the list of chunks
                chunks.append(chunk_text)
                # Reset the current chunk and token count
                current_chunk = []
                current_chunk_tokens = 0
            else:
                # If a single sentence exceeds max_tokens, handle it separately
                print(f"Warning: A single sentence exceeds max_tokens limit: {sentence[:50]}...")
                # Split the long sentence into smaller parts
                sub_sentences = split_long_sentence(sentence, max_tokens)
                # Add the sub-sentences to the chunks
                chunks.extend(sub_sentences)
                # Move to the next sentence
                i += 1
                # Reset the current chunk and token count
                current_chunk = []
                current_chunk_tokens = 0

    # Add any remaining sentences as the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk).strip()
        chunk_text = chunk_text.strip('.').strip()
        chunks.append(chunk_text)

    # Return the list of chunks
    return chunks


def split_long_sentence(sentence, max_tokens):
    """
    Splits a long sentence into smaller chunks based on punctuation.

    Parameters:
    - sentence (str): The sentence to split.
    - max_tokens (int): Maximum number of tokens per chunk.

    Returns:
    - sub_chunks (list): A list of smaller sentence chunks.
    """
    # Initialize a list to store the sub-chunks
    sub_chunks = []
    # Define punctuation marks to split on
    punctuation = [".", ",", ";", ":", "-", "–", "—", "(", ")", "\"", "'"]
    # Initialize variables
    sub_sentences = []
    current_sub = ""
    # Iterate over each character in the sentence
    for char in sentence:
        current_sub += char
        # If the character is a punctuation mark, split here
        if char in punctuation:
            sub_sentences.append(current_sub.strip())
            current_sub = ""
    # Add any remaining text as a sub-sentence
    if current_sub:
        sub_sentences.append(current_sub.strip())

    # Further split the sub-sentences if necessary
    for sub in sub_sentences:
        # Tokenize the sub-sentence
        tokens = tokenizer.encode(sub, add_special_tokens=False)
        # If the sub-sentence is within the token limit, add it to sub_chunks
        if len(tokens) <= max_tokens:
            sub_chunks.append(sub)
        else:
            # Split the sub-sentence by whitespace as a last resort
            words = sub.split()
            temp = ""
            for word in words:
                # Tokenize the word
                word_tokens = tokenizer.encode(word, add_special_tokens=False)
                # Check if adding the word exceeds the max_tokens limit
                if len(tokenizer.encode(temp + " " + word, add_special_tokens=False)) <= max_tokens:
                    temp = temp + " " + word if temp else word
                else:
                    # Add the accumulated words as a sub-chunk
                    if temp:
                        sub_chunks.append(temp)
                    temp = word
            # Add any remaining words as a sub-chunk
            if temp:
                sub_chunks.append(temp)
    # Return the list of sub-chunks
    return sub_chunks


def read_and_split_documents(folder_path):
    """
    Reads all .txt files from the specified folder and splits them into chunks.

    Parameters:
    - folder_path (str): The path to the folder containing .txt files.

    Returns:
    - documents (dict): A dictionary mapping file paths to lists of text chunks.
    """
    # Initialize a dictionary to store documents and their chunks
    documents = {}
    # Get a list of all .txt files in the folder
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    print(f"Found {len(txt_files)} .txt files: {txt_files}")
    # Check if any .txt files were found
    if len(txt_files) == 0:
        print("No .txt files found in the specified folder.")
    # Loop through each file
    for file_path in txt_files:
        print(f"\nProcessing file: {file_path}")
        try:
            # Open and read the file with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Check if the file is empty
                if not text.strip():
                    print(f"Warning: The file {file_path} is empty.")
                else:
                    # Split the text into chunks
                    chunks = split_text_into_chunks(text, max_tokens)
                    print(f"Number of chunks created from {file_path}: {len(chunks)}")
                    # Store the chunks in the documents dictionary
                    documents[file_path] = chunks
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    # Return the documents dictionary
    return documents


def initialize_qdrant(client, collection_name, vector_size, distance_metric="Cosine", recreate=False):
    """
    Initializes a Qdrant collection. Creates it if it doesn't exist or recreates it if specified.

    Parameters:
    - client (QdrantClient): The Qdrant client instance.
    - collection_name (str): The name of the collection.
    - vector_size (int): The dimension of the vector embeddings.
    - distance_metric (str): The distance metric to use ('Cosine', 'Euclidean', etc.).
    - recreate (bool): If True, deletes the existing collection and creates a new one.

    Returns:
    - None
    """
    # Retrieve the list of existing collections
    existing_collections = client.get_collections().collections
    # Check if the collection already exists
    if collection_name in [col.name for col in existing_collections]:
        if recreate:
            print(f"Recreating Qdrant collection '{collection_name}'.")
            # Delete the existing collection
            client.delete_collection(collection_name=collection_name)
            print(f"Deleted existing collection '{collection_name}'.")
            # Create a new collection with the specified parameters
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=distance_metric
                )
            )
            print(f"Created new collection '{collection_name}'.")
        else:
            print(f"Qdrant collection '{collection_name}' already exists.")
    else:
        print(f"Creating Qdrant collection '{collection_name}' with vector size {vector_size} and distance metric '{distance_metric}'.")
        # Create the collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=distance_metric
            )
        )


def upsert_embeddings_qdrant(client, collection_name, embeddings, metadata, payload_fields):
    """
    Upserts embeddings and their metadata into a Qdrant collection with valid point IDs.

    Parameters:
    - client (QdrantClient): The Qdrant client instance.
    - collection_name (str): The name of the collection.
    - embeddings (list or np.ndarray): The list or array of embedding vectors.
    - metadata (list of dict): The list of metadata dictionaries corresponding to each embedding.
    - payload_fields (list): The list of metadata fields to store in Qdrant.

    Returns:
    - None
    """
    # Initialize a list to store the points to upsert
    points = []
    # Loop through each embedding and its metadata
    for idx, (meta, vector) in enumerate(zip(metadata, embeddings)):
        # Use an unsigned integer as the point ID
        point_id = idx  # or idx + 1 if you want to start from 1

        # Include the unique identifier in the payload
        unique_id = f"{meta['document']}_{meta['passage_index']}"
        payload = {field: meta[field] for field in payload_fields}
        payload['unique_id'] = unique_id  # Add the unique ID to the payload

        # Convert the embedding to a list if it's a NumPy array
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        elif isinstance(vector, list):
            vector = vector
        else:
            vector = list(vector)  # Fallback to list conversion

        # Create a point structure with the embedding and payload
        points.append(qmodels.PointStruct(id=point_id, vector=vector, payload=payload))

    # Upsert the points into the Qdrant collection
    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"Upserted {len(points)} points into Qdrant collection '{collection_name}'.")
    else:
        print("No valid points to upsert.")


def perform_similarity_search(client, collection_name, query_embedding, top_k=5):
    """
    Performs a similarity search in Qdrant and retrieves the top_k most similar passages.

    Parameters:
    - client (QdrantClient): The Qdrant client instance.
    - collection_name (str): The name of the collection.
    - query_embedding (list or np.ndarray): The embedding vector for the query.
    - top_k (int): The number of top similar passages to retrieve.

    Returns:
    - results (list): A list of dictionaries containing similar passages and their metadata.
    """
    # Ensure the query_embedding is a list
    if isinstance(query_embedding, np.ndarray):
        query_vector = query_embedding.tolist()
    elif isinstance(query_embedding, list):
        query_vector = query_embedding
    else:
        query_vector = query_embedding.cpu().numpy().tolist()

    # Perform a search in Qdrant using the query embedding
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )
    # Initialize a list to store the results
    results = []
    # Loop through each search result
    for hit in search_result:
        result = {
            'id': hit.id,  # Point ID as an integer
            'score': hit.score,
            'passage': hit.payload.get('passage'),
            'document': hit.payload.get('document'),
            'passage_index': hit.payload.get('passage_index'),
            'cluster': hit.payload.get('cluster'),
            'unique_id': hit.payload.get('unique_id')  # Retrieve the unique ID from payload
        }
        results.append(result)
    # Return the list of results
    return results


def highlight_similar_text_using_model(query, passage, model, stopwords_set, threshold=0.7):
    """
    Highlights similar content words in the passage based on semantic similarity with the query.

    Parameters:
    - query (str): The query text.
    - passage (str): The passage text.
    - model (SentenceTransformer): The pre-trained model for embeddings.
    - stopwords_set (set): Set of stopwords to exclude.
    - threshold (float): Similarity threshold for highlighting words.

    Returns:
    - highlighted_passage (str): The passage with similar words highlighted.
    """
    # Tokenize query and passage into words
    query_tokens = re.findall(r'\b\w+\b', query.lower())
    passage_tokens = passage.split()

    # Filter out stopwords
    query_tokens_filtered = [word for word in query_tokens if word not in stopwords_set]
    passage_tokens_filtered = [word for word in passage_tokens if re.sub(r'[^\w\s]', '', word).lower() not in stopwords_set]

    # Get unique passage tokens
    unique_passage_tokens = list(set(passage_tokens_filtered))

    if not query_tokens_filtered or not unique_passage_tokens:
        # If there are no tokens to compare, return the original passage
        return html.escape(passage)

    # Encode the query and passage tokens
    query_embeddings = model.encode(query_tokens_filtered, convert_to_tensor=True)
    passage_embeddings = model.encode(unique_passage_tokens, convert_to_tensor=True)

    # Move tensors to CPU if necessary
    query_embeddings = query_embeddings.cpu()
    passage_embeddings = passage_embeddings.cpu()

    # Compute cosine similarities
    similarities = cosine_similarity(query_embeddings.numpy(), passage_embeddings.numpy())

    # Find tokens with similarity above the threshold
    similar_tokens = set()
    for i, query_word in enumerate(query_tokens_filtered):
        for j, passage_word in enumerate(unique_passage_tokens):
            if similarities[i][j] >= threshold:
                similar_tokens.add(passage_word.lower())

    # Highlight similar tokens in the passage
    highlighted_words = []
    for word in passage_tokens:
        word_clean = re.sub(r'[^\w\s]', '', word)
        if word_clean.lower() in similar_tokens:
            highlighted_word = f'<mark>{html.escape(word)}</mark>'
        else:
            highlighted_word = html.escape(word)
        highlighted_words.append(highlighted_word)

    highlighted_passage = ' '.join(highlighted_words)
    return highlighted_passage


def retrieve_and_verify_qdrant_data(client, collection_name, original_metadata):
    """
    Retrieves all data from Qdrant and verifies passage indices.

    Parameters:
    - client (QdrantClient): The Qdrant client instance.
    - collection_name (str): The name of the collection.
    - original_metadata (list of dict): The original metadata used during upsert.

    Returns:
    - None
    """
    # Initialize the offset for pagination
    offset = None
    all_hits = []

    while True:
        # Corrected parameter name from 'scroll' to 'offset'
        result, next_page_offset = client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=100,
            with_payload=True
        )
        all_hits.extend(result)

        if next_page_offset is None:
            break
        offset = next_page_offset

    # Create a mapping from unique_id to payload
    qdrant_data = {}
    for hit in all_hits:
        qdrant_data[hit.payload['unique_id']] = hit.payload

    # Verify each entry
    mismatches = []
    for meta in original_metadata:
        unique_id = f"{meta['document']}_{meta['passage_index']}"
        if unique_id not in qdrant_data:
            mismatches.append(f"Missing in Qdrant: {unique_id}")
            continue
        qdrant_passage = qdrant_data[unique_id]['passage']
        original_passage = meta['passage']
        if qdrant_passage != original_passage:
            mismatches.append(f"Mismatch in passage for {unique_id}")

    if mismatches:
        print("Found mismatches in Qdrant data:")
        for mismatch in mismatches:
            print(mismatch)
    else:
        print("All passages in Qdrant match the original metadata.")


# ------------------------ Main Pipeline ------------------------

def main():
    # Read and split the documents into chunks
    documents = read_and_split_documents(folder_path)

    # Flatten the documents into a list of passages and collect metadata
    passages = []
    metadata = []
    # Loop through each document and its chunks
    for doc_name, chunks in documents.items():
        for idx, chunk in enumerate(chunks):
            # Add the chunk to the passages list
            passages.append(chunk)
            # Collect metadata for the chunk
            metadata.append({
                'document': os.path.basename(doc_name),
                'passage_index': idx,
                'passage': chunk  # Ensure 'passage' is included
            })

    print(f"\nTotal number of passages: {len(passages)}")

    # Check if any passages were extracted
    if len(passages) == 0:
        print("No passages were extracted. Please check the document reading and splitting process.")
        return

    # Encode the passages into embeddings using the model
    print("\nEncoding passages into embeddings...")
    embeddings = model.encode(passages, convert_to_tensor=False, show_progress_bar=True)
    # Convert embeddings to a NumPy array for clustering
    clustering_data = np.array(embeddings)

    # Determine the optimal number of clusters using the Elbow Method
    print("\nDetermining the optimal number of clusters using the Elbow Method...")
    distortions = []
    # Set the maximum number of clusters to try
    max_k = min(30, len(passages) - 1)  # Adjust max_k based on the number of passages
    # Create a range of cluster counts to try
    K = range(1, max_k + 1)

    # Loop through each value of k
    for k in K:
        # Initialize KMeans with k clusters
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        # Fit the model to the data
        kmeans_model.fit(clustering_data)
        # Append the inertia (sum of squared distances) to the distortions list
        distortions.append(kmeans_model.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 4))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances (Inertia)')
    plt.title('Elbow Method for Optimal k')

    # Use KneeLocator to find the elbow point
    try:
        # Initialize the KneeLocator
        knee_locator = KneeLocator(
            K, distortions, curve='convex', direction='decreasing',
            interp_method='interp1d', polynomial_degree=4
        )
        # Get the elbow point
        optimal_clusters_elbow = knee_locator.elbow

        # If an elbow point was detected, annotate it on the plot
        if optimal_clusters_elbow:
            plt.vlines(optimal_clusters_elbow, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='red')
            plt.text(optimal_clusters_elbow + 0.5, distortions[optimal_clusters_elbow - 1], f'k={optimal_clusters_elbow}', color='red')
            print(f"Optimal number of clusters detected by Elbow Method: {optimal_clusters_elbow}")
        else:
            # If no elbow point was detected, prompt the user to input it
            print("Elbow point not detected automatically.")
            optimal_clusters_elbow = int(input("Enter the optimal number of clusters based on the Elbow Method plot: "))
    except Exception as e:
        # Handle any errors during elbow detection
        print(f"An error occurred while detecting the elbow point: {e}")
        optimal_clusters_elbow = int(input("Enter the optimal number of clusters based on the Elbow Method plot: "))

    # Save the elbow plot to a file
    plt.savefig('elbow_method.png')
    # Display the elbow plot
    plt.show()

    # Perform KMeans clustering with the optimal number of clusters
    final_k = optimal_clusters_elbow
    print(f"\nClustering embeddings into {final_k} clusters...")
    # Initialize KMeans with the final number of clusters
    clustering_model = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    # Fit the model to the data
    clustering_model.fit(clustering_data)
    # Get the cluster assignments for each data point
    cluster_assignments = clustering_model.labels_

    # Update metadata with cluster assignments
    for idx, label in enumerate(cluster_assignments):
        metadata[idx]['cluster'] = int(label)

    # Map clusters to passages for easy access
    clusters = {}
    for idx, label in enumerate(cluster_assignments):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({
            'passage': passages[idx],
            'document': metadata[idx]['document'],
            'passage_index': metadata[idx]['passage_index']
        })

    # Write the clusters to a CSV file for inspection
    output_csv = 'clusters.csv'
    print(f"\nWriting clusters to {output_csv}...")
    with open(output_csv, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['Cluster', 'Document', 'Passage Index', 'Passage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Loop through each cluster and its passages
        for cluster_id, passages_in_cluster in clusters.items():
            for passage_info in passages_in_cluster:
                writer.writerow({
                    'Cluster': cluster_id,
                    'Document': passage_info['document'],
                    'Passage Index': passage_info['passage_index'],
                    'Passage': passage_info['passage']
                })

    print("Clusters have been successfully written to the CSV file.")

    # ------------------------ Visualization ------------------------

    # Visualize the clusters using t-SNE
    try:
        print("\nGenerating t-SNE visualization...")
        num_samples = len(embeddings)

        # Check if there are enough samples for t-SNE
        if num_samples < 3:
            print("Not enough samples for t-SNE visualization.")
        else:
            # Set perplexity parameter based on the number of samples
            perplexity = min(30, num_samples - 1)
            print(f"Using perplexity={perplexity}")

            # Initialize t-SNE with adjusted parameters
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                n_iter=1000,
                learning_rate='auto'  # Adjust learning rate automatically
            )
            # Fit t-SNE to the data and reduce dimensions
            embeddings_2d = tsne.fit_transform(clustering_data)

            # Create a DataFrame for plotting
            df = pd.DataFrame()
            df['x'] = embeddings_2d[:, 0]
            df['y'] = embeddings_2d[:, 1]
            df['Cluster'] = cluster_assignments.astype(str)
            df['Document'] = [m['document'] for m in metadata]

            # Plot the clusters using seaborn
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=df,
                x='x',
                y='y',
                hue='Cluster',
                style='Document',
                palette='tab10',
                s=100
            )
            plt.title('t-SNE Visualization of Clusters')
            plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            # Save the plot to a file
            plt.savefig('tsne_clusters.png')  # Saves the plot as a PNG file

            # Display the plot
            plt.show()
    except ImportError:
        print("t-SNE visualization requires scikit-learn and seaborn. Visualization skipped.")
    except ValueError as e:
        print(f"Error during t-SNE visualization: {e}")
        print("Adjust the perplexity parameter to be less than the number of samples.")

    # ------------------------ Qdrant Integration ------------------------

    # Initialize Qdrant client for vector database operations
    print("\nInitializing Qdrant client...")
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Determine the vector size from the embeddings
    vector_size = len(embeddings[0])

    # Initialize Qdrant collection
    initialize_qdrant(
        client,
        qdrant_collection_name,
        vector_size,
        distance_metric="Cosine",
        recreate=False  # Set to True if you want to recreate the collection
    )

    # Prepare payload fields to store in Qdrant
    payload_fields = ['passage', 'document', 'passage_index', 'cluster']

    # Upsert embeddings and metadata into Qdrant
    print("\nUpserting embeddings and metadata into Qdrant...")
    upsert_embeddings_qdrant(
        client=client,
        collection_name=qdrant_collection_name,
        embeddings=embeddings,
        metadata=metadata,
        payload_fields=payload_fields
    )

    # Verify passage indices in Qdrant
    print("\nVerifying passage indices in Qdrant...")
    retrieve_and_verify_qdrant_data(
        client=client,
        collection_name=qdrant_collection_name,
        original_metadata=metadata
    )

    # ------------------------ Similarity Search with Model-based Highlighting ------------------------

    try:
        print("\nPerforming similarity searches for provided phrases and highlighting similar text...")
        # List of queries in Greek
        queries = [
            "Πότε πρέπει να υποβάλω αίτηση για άδεια μητρότητας ή πατρότητας;",
            "Ποιοι κανόνες ασφάλειας και υγείας πρέπει να τηρούνται στον εργασιακό χώρο;",
            "Πώς αξιολογείται η απόδοσή μου και ποια είναι τα κριτήρια;",
            "Ποιες ευκαιρίες εκπαίδευσης και ανάπτυξης παρέχει η εταιρεία;",
            "Τι περιλαμβάνει η πολιτική εργασιακής ισορροπίας;",
            "Ποια προγράμματα υποστήριξης προσφέρονται για την υγεία και την ευεξία των εργαζομένων;"
        ]

        # Create an HTML report
        report_filename = 'similarity_report.html'
        with open(report_filename, 'w', encoding='utf-8') as report_file:
            # Write the header of the HTML file
            report_file.write('<html><head><meta charset="UTF-8"><title>Similarity Report</title></head><body>')
            report_file.write('<h1>Similarity Report</h1>')

            # Loop through each query
            for query in queries:
                report_file.write(f'<h2>Query: {html.escape(query)}</h2>')
                print(f"\nQuery: {query}")
                # Encode the query into an embedding
                query_embedding = model.encode([query], convert_to_tensor=False)[0]
                # Ensure the query embedding is on CPU
                if hasattr(query_embedding, 'cpu'):
                    query_embedding = query_embedding.cpu().numpy()
                else:
                    query_embedding = np.array(query_embedding)

                # Perform similarity search in Qdrant
                similar_passages = perform_similarity_search(
                    client=client,
                    collection_name=qdrant_collection_name,
                    query_embedding=query_embedding,
                    top_k=3
                )

                # Check if any similar passages were found
                if similar_passages:
                    report_file.write('<ol>')
                    print("\nTop 3 similar passages:")
                    # Loop through each similar passage
                    for idx, passage in enumerate(similar_passages, 1):
                        # Highlight similar text using the model and stopwords
                        highlighted_passage = highlight_similar_text_using_model(
                            query,
                            passage['passage'],
                            model,
                            greek_stopwords,
                            threshold=0.7  # Increased threshold
                        )

                        # Write to the HTML report
                        report_file.write(f'<li>')
                        report_file.write(f'<p><strong>Rank {idx}</strong></p>')
                        report_file.write(f'<p><strong>Score:</strong> {passage["score"]:.4f}</p>')
                        report_file.write(f'<p><strong>Document:</strong> {html.escape(passage["document"])}</p>')
                        report_file.write(f'<p><strong>Passage Index:</strong> {passage["passage_index"]}</p>')
                        report_file.write(f'<p><strong>Passage:</strong> {highlighted_passage}</p>')
                        report_file.write('</li>')

                        # Also print the results to the console
                        print(f"\nRank {idx}:")
                        print(f"Score: {passage['score']:.4f}")
                        print(f"Document: {passage['document']}")
                        print(f"Passage Index: {passage['passage_index']}")
                        print(f"Passage: {passage['passage']}")
                else:
                    report_file.write('<p>No similar passages found.</p>')
                    print("No similar passages found.")

                report_file.write('</ol>')

            # Write the footer of the HTML file
            report_file.write('</body></html>')

        print(f"\nSimilarity report generated: {report_filename}")

    except Exception as e:
        print(f"An error occurred during similarity search: {e}")#test


# ------------------------ Execute Pipeline ------------------------

if __name__ == "__main__":
    main()
