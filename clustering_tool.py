# ------------------------ Import Necessary Libraries ------------------------

import os  # For interacting with the operating system
import glob  # For file pattern matching
import re  # For regex-based text splitting
from tqdm import tqdm  # For progress bars
from sentence_transformers import SentenceTransformer  # For generating embeddings
from sklearn.cluster import KMeans  # For clustering embeddings
from sklearn.manifold import TSNE  # For dimensionality reduction in visualization
from transformers import AutoTokenizer  # For tokenizing text
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For statistical data visualization
import csv  # For writing CSV files
from kneed import KneeLocator  # For detecting the 'elbow' in the elbow method
from qdrant_client import QdrantClient  # For interacting with Qdrant vector database
from qdrant_client.http import models as qmodels  # For Qdrant data models
import html  # For escaping HTML content in strings
from sklearn.metrics.pairwise import cosine_similarity  # For computing similarity between vectors
import nltk  # For natural language processing tasks
from nltk.corpus import stopwords  # For accessing stopwords

# ------------------------ NLTK Setup ------------------------

# Download the 'stopwords' dataset from NLTK if not already present
nltk.download('stopwords')

# ------------------------ Configuration ------------------------

# Define the path to the folder containing the .txt files to process
folder_path = 'data'

# Define model parameters
max_tokens = 128  # Maximum number of tokens per text chunk, aligned with the model's max sequence length

# Initialize the tokenizer associated with the pre-trained embedding model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Initialize the SentenceTransformer model for generating text embeddings
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Qdrant Configuration for vector database connection
qdrant_host = 'localhost'  # Host address for Qdrant
qdrant_port = 6333         # Port number for Qdrant
qdrant_collection_name = 'text_clusters'  # Name of the collection in Qdrant to store vectors

# Get Greek stopwords from NLTK and create a set for faster lookup
greek_stopwords = set(stopwords.words('greek'))

# Add additional common Greek words to the stopwords set
additional_stopwords = {
    'ένα', 'πριν', 'από', 'προς', 'τους', 'στην', 'στις', 'της', 'του', 'μεταξύ', 'ή', 'και'
}
greek_stopwords.update(additional_stopwords)

# ------------------------ Function Definitions ------------------------

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
    # Use regex to split the text into sentences at punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []  # List to store the resulting chunks
    current_chunk = []  # List to accumulate sentences for the current chunk
    current_chunk_tokens = 0  # Counter for tokens in the current chunk

    i = 0  # Index for iterating over sentences
    while i < len(sentences):
        sentence = sentences[i].strip()  # Get the current sentence and strip whitespace
        # Tokenize the sentence without adding special tokens
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_token_count = len(sentence_tokens)  # Count the tokens in the sentence

        # Check if adding the sentence exceeds the max_tokens limit
        if current_chunk_tokens + sentence_token_count <= max_tokens:
            current_chunk.append(sentence)  # Add sentence to the current chunk
            current_chunk_tokens += sentence_token_count  # Update token count
            i += 1  # Move to the next sentence
        else:
            if current_chunk:
                # Finalize the current chunk by joining sentences
                chunk_text = ' '.join(current_chunk).strip()
                chunk_text = chunk_text.strip('.').strip()  # Remove leading/trailing periods and whitespace
                chunks.append(chunk_text)  # Add the chunk to the list
                current_chunk = []  # Reset the current chunk
                current_chunk_tokens = 0  # Reset the token count
            else:
                # Handle single sentences that exceed max_tokens
                print(f"Warning: A single sentence exceeds max_tokens limit: {sentence[:50]}...")
                # Split the long sentence into smaller parts
                sub_sentences = split_long_sentence(sentence, max_tokens)
                chunks.extend(sub_sentences)  # Add sub-sentences to the chunks list
                i += 1  # Move to the next sentence
                current_chunk = []  # Reset the current chunk
                current_chunk_tokens = 0  # Reset the token count

    # Add any remaining sentences as the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk).strip()
        chunk_text = chunk_text.strip('.').strip()
        chunks.append(chunk_text)

    return chunks  # Return the list of chunks

def split_long_sentence(sentence, max_tokens):
    """
    Splits a long sentence into smaller chunks based on punctuation.

    Parameters:
    - sentence (str): The sentence to split.
    - max_tokens (int): Maximum number of tokens per chunk.

    Returns:
    - sub_chunks (list): A list of smaller sentence chunks.
    """
    sub_chunks = []  # List to store the sub-chunks
    # Define punctuation marks to split on
    punctuation = [".", ",", ";", ":", "-", "–", "—", "(", ")", "\"", "'"]
    sub_sentences = []  # List to store sub-sentences
    current_sub = ""  # Current substring being built

    # Iterate over each character in the sentence
    for char in sentence:
        current_sub += char  # Add character to current substring
        if char in punctuation:
            # If character is punctuation, split here
            sub_sentences.append(current_sub.strip())
            current_sub = ""  # Reset current substring
    if current_sub:
        sub_sentences.append(current_sub.strip())  # Add any remaining text

    # Further split the sub-sentences if necessary
    for sub in sub_sentences:
        tokens = tokenizer.encode(sub, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            sub_chunks.append(sub)  # Add sub-sentence to sub-chunks
        else:
            # Split the sub-sentence by words if still too long
            words = sub.split()
            temp = ""  # Temporary string to build up words
            for word in words:
                # Check if adding the word exceeds max_tokens
                if len(tokenizer.encode(temp + " " + word, add_special_tokens=False)) <= max_tokens:
                    temp = temp + " " + word if temp else word
                else:
                    if temp:
                        sub_chunks.append(temp)  # Add accumulated words as a sub-chunk
                    temp = word  # Start a new sub-chunk with the current word
            if temp:
                sub_chunks.append(temp)  # Add any remaining words as a sub-chunk
    return sub_chunks  # Return the list of sub-chunks

def read_and_split_documents(folder_path):
    """
    Reads all .txt files from the specified folder and splits them into chunks.

    Parameters:
    - folder_path (str): The path to the folder containing .txt files.

    Returns:
    - documents (dict): A dictionary mapping file paths to lists of text chunks.
    """
    documents = {}  # Dictionary to store file paths and their chunks
    # Get a list of all .txt files in the folder
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    print(f"Found {len(txt_files)} .txt files: {txt_files}")
    if len(txt_files) == 0:
        print("No .txt files found in the specified folder.")

    # Process each file
    for file_path in txt_files:
        print(f"\nProcessing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()  # Read the entire file content
                if not text.strip():
                    print(f"Warning: The file {file_path} is empty.")
                else:
                    # Split the text into chunks
                    chunks = split_text_into_chunks(text, max_tokens)
                    print(f"Number of chunks created from {file_path}: {len(chunks)}")
                    documents[file_path] = chunks  # Add chunks to the documents dictionary
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    return documents  # Return the documents dictionary

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
            client.delete_collection(collection_name=collection_name)  # Delete existing collection
            print(f"Deleted existing collection '{collection_name}'.")
            # Create a new collection with specified vector size and distance metric
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
        # Create a new collection if it doesn't exist
        print(f"Creating Qdrant collection '{collection_name}' with vector size {vector_size} and distance metric '{distance_metric}'.")
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
    points = []  # List to store the points to be upserted
    for idx, (meta, vector) in enumerate(zip(metadata, embeddings)):
        point_id = idx  # Unique point ID (can be adjusted if needed)

        # Create a unique identifier combining document name and passage index
        unique_id = f"{meta['document']}_{meta['passage_index']}"
        # Prepare the payload with specified fields and unique ID
        payload = {field: meta[field] for field in payload_fields}
        payload['unique_id'] = unique_id  # Add unique ID to payload

        # Ensure the embedding vector is in list format
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        elif isinstance(vector, list):
            vector = vector
        else:
            vector = list(vector)  # Convert to list as a fallback

        # Create a PointStruct object with the point ID, vector, and payload
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

    results = []  # List to store search results
    for hit in search_result:
        result = {
            'id': hit.id,  # Point ID
            'score': hit.score,  # Similarity score
            'passage': hit.payload.get('passage'),  # Retrieved passage text
            'document': hit.payload.get('document'),  # Document name
            'passage_index': hit.payload.get('passage_index'),  # Index of the passage in the document
            'cluster': hit.payload.get('cluster'),  # Cluster assignment
            'unique_id': hit.payload.get('unique_id')  # Unique identifier
        }
        results.append(result)
    return results  # Return the list of results

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

    # Filter out stopwords from query and passage tokens
    query_tokens_filtered = [word for word in query_tokens if word not in stopwords_set]
    passage_tokens_filtered = [word for word in passage_tokens if re.sub(r'[^\w\s]', '', word).lower() not in stopwords_set]

    # Get unique tokens from the passage
    unique_passage_tokens = list(set(passage_tokens_filtered))

    # Return the original passage if no tokens are left after filtering
    if not query_tokens_filtered or not unique_passage_tokens:
        return html.escape(passage)

    # Encode the tokens using the model
    query_embeddings = model.encode(query_tokens_filtered, convert_to_tensor=True)
    passage_embeddings = model.encode(unique_passage_tokens, convert_to_tensor=True)

    # Move embeddings to CPU if necessary
    query_embeddings = query_embeddings.cpu()
    passage_embeddings = passage_embeddings.cpu()

    # Compute cosine similarities between query and passage tokens
    similarities = cosine_similarity(query_embeddings.numpy(), passage_embeddings.numpy())

    # Identify tokens in the passage that are similar to the query tokens
    similar_tokens = set()
    for i, query_word in enumerate(query_tokens_filtered):
        for j, passage_word in enumerate(unique_passage_tokens):
            if similarities[i][j] >= threshold:
                similar_tokens.add(passage_word.lower())

    # Highlight similar tokens in the passage
    highlighted_words = []
    for word in passage_tokens:
        word_clean = re.sub(r'[^\w\s]', '', word)  # Remove punctuation
        if word_clean.lower() in similar_tokens:
            highlighted_word = f'<mark>{html.escape(word)}</mark>'  # Highlight word
        else:
            highlighted_word = html.escape(word)  # Escape HTML characters
        highlighted_words.append(highlighted_word)

    highlighted_passage = ' '.join(highlighted_words)  # Reconstruct the passage
    return highlighted_passage  # Return the highlighted passage

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
    offset = None  # For pagination
    all_hits = []  # List to store all retrieved points

    while True:
        # Retrieve points from Qdrant with pagination
        result, next_page_offset = client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=100,
            with_payload=True
        )
        all_hits.extend(result)  # Add retrieved points to the list

        if next_page_offset is None:
            break  # Exit if no more points to retrieve
        offset = next_page_offset  # Update offset for next iteration

    # Create a mapping from unique_id to payload
    qdrant_data = {}
    for hit in all_hits:
        qdrant_data[hit.payload['unique_id']] = hit.payload

    mismatches = []  # List to store any mismatches found
    for meta in original_metadata:
        unique_id = f"{meta['document']}_{meta['passage_index']}"
        if unique_id not in qdrant_data:
            mismatches.append(f"Missing in Qdrant: {unique_id}")
            continue
        qdrant_passage = qdrant_data[unique_id]['passage']
        original_passage = meta['passage']
        if qdrant_passage != original_passage:
            mismatches.append(f"Mismatch in passage for {unique_id}")

    # Report mismatches or confirm data integrity
    if mismatches:
        print("Found mismatches in Qdrant data:")
        for mismatch in mismatches:
            print(mismatch)
    else:
        print("All passages in Qdrant match the original metadata.")

# ------------------------ Main Pipeline ------------------------

def main():
    # ------------------------ Step 1: Read and Split Documents ------------------------

    # Read all .txt files and split them into chunks
    documents = read_and_split_documents(folder_path)

    # Flatten the documents into a list of passages and collect metadata
    passages = []  # List to store all passages
    metadata = []  # List to store metadata for each passage
    for doc_name, chunks in documents.items():
        for idx, chunk in enumerate(chunks):
            passages.append(chunk)  # Add the passage to the list
            # Collect metadata for the passage
            metadata.append({
                'document': os.path.basename(doc_name),  # Document name without the path
                'passage_index': idx,  # Index of the passage in the document
                'passage': chunk  # The passage text
            })

    print(f"\nTotal number of passages: {len(passages)}")

    # Check if any passages were extracted
    if len(passages) == 0:
        print("No passages were extracted. Please check the document reading and splitting process.")
        return  # Exit the program if no passages

    # ------------------------ Step 2: Encode Passages ------------------------

    # Encode the passages into embeddings using the model
    print("\nEncoding passages into embeddings...")
    embeddings = model.encode(passages, convert_to_tensor=False, show_progress_bar=True)
    # Convert embeddings to a NumPy array for clustering
    clustering_data = np.array(embeddings)

    # ------------------------ Step 3: Determine Optimal Number of Clusters ------------------------

    print("\nDetermining the optimal number of clusters using the Elbow Method...")
    distortions = []  # List to store distortion values
    # Set the maximum number of clusters to try
    max_k = min(30, len(passages) - 1)  # Adjust max_k based on the number of passages
    K = range(1, max_k + 1)  # Range of k values to try

    for k in K:
        # Initialize KMeans with k clusters
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_model.fit(clustering_data)  # Fit the model to the data
        distortions.append(kmeans_model.inertia_)  # Append the inertia to distortions

    # Plot the elbow curve
    plt.figure(figsize=(8, 4))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances (Inertia)')
    plt.title('Elbow Method for Optimal k')

    # Use KneeLocator to find the elbow point
    try:
        knee_locator = KneeLocator(
            K, distortions, curve='convex', direction='decreasing',
            interp_method='interp1d', polynomial_degree=4
        )
        optimal_clusters_elbow = knee_locator.elbow  # Optimal k

        if optimal_clusters_elbow:
            # Annotate the elbow point on the plot
            plt.vlines(optimal_clusters_elbow, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='red')
            plt.text(optimal_clusters_elbow + 0.5, distortions[optimal_clusters_elbow - 1],
                     f'k={optimal_clusters_elbow}', color='red')
            print(f"Optimal number of clusters detected by Elbow Method: {optimal_clusters_elbow}")
        else:
            # Prompt user input if elbow point not detected
            print("Elbow point not detected automatically.")
            optimal_clusters_elbow = int(input("Enter the optimal number of clusters based on the Elbow Method plot: "))
    except Exception as e:
        # Handle errors during elbow detection
        print(f"An error occurred while detecting the elbow point: {e}")
        optimal_clusters_elbow = int(input("Enter the optimal number of clusters based on the Elbow Method plot: "))

    # Save and display the elbow plot
    plt.savefig('elbow_method.png')
    plt.show()

    # ------------------------ Step 4: Perform KMeans Clustering ------------------------

    final_k = optimal_clusters_elbow  # Use the optimal number of clusters
    print(f"\nClustering embeddings into {final_k} clusters...")
    # Initialize KMeans with the final number of clusters
    clustering_model = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    clustering_model.fit(clustering_data)  # Fit the model
    cluster_assignments = clustering_model.labels_  # Get cluster labels

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

    # ------------------------ Step 5: Write Clusters to CSV ------------------------

    output_csv = 'clusters.csv'
    print(f"\nWriting clusters to {output_csv}...")
    with open(output_csv, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['Cluster', 'Document', 'Passage Index', 'Passage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for cluster_id, passages_in_cluster in clusters.items():
            for passage_info in passages_in_cluster:
                writer.writerow({
                    'Cluster': cluster_id,
                    'Document': passage_info['document'],
                    'Passage Index': passage_info['passage_index'],
                    'Passage': passage_info['passage']
                })

    print("Clusters have been successfully written to the CSV file.")

    # ------------------------ Step 6: Visualization ------------------------

    # Visualize the clusters using t-SNE
    try:
        print("\nGenerating t-SNE visualization...")
        num_samples = len(embeddings)

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
            embeddings_2d = tsne.fit_transform(clustering_data)  # Reduce dimensions

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

            # Save and display the plot
            plt.savefig('tsne_clusters.png')
            plt.show()
    except ImportError:
        print("t-SNE visualization requires scikit-learn and seaborn. Visualization skipped.")
    except ValueError as e:
        print(f"Error during t-SNE visualization: {e}")
        print("Adjust the perplexity parameter to be less than the number of samples.")

    # ------------------------ Step 7: Qdrant Integration ------------------------

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

    # ------------------------ Step 8: Verify Data in Qdrant ------------------------

    print("\nVerifying passage indices in Qdrant...")
    retrieve_and_verify_qdrant_data(
        client=client,
        collection_name=qdrant_collection_name,
        original_metadata=metadata
    )

    # ------------------------ Step 9: Similarity Search and Highlighting ------------------------

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

        # Create an HTML report to display results
        report_filename = 'similarity_report.html'
        with open(report_filename, 'w', encoding='utf-8') as report_file:
            # Write the header of the HTML file
            report_file.write('<html><head><meta charset="UTF-8"><title>Similarity Report</title></head><body>')
            report_file.write('<h1>Similarity Report</h1>')

            # Process each query
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

                if similar_passages:
                    report_file.write('<ol>')
                    print("\nTop 3 similar passages:")
                    # Process each similar passage
                    for idx, passage in enumerate(similar_passages, 1):
                        # Highlight similar text using the model and stopwords
                        highlighted_passage = highlight_similar_text_using_model(
                            query,
                            passage['passage'],
                            model,
                            greek_stopwords,
                            threshold=0.7  # Similarity threshold
                        )

                        # Write the result to the HTML report
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
        print(f"An error occurred during similarity search: {e}")

# ------------------------ Execute Pipeline ------------------------

if __name__ == "__main__":
    main()
