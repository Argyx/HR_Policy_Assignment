# Import necessary libraries
import os
import glob
import spacy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from kneed import KneeLocator
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Ensure plots are rendered inline if using Jupyter Notebook
# %matplotlib inline  # Uncomment if using Jupyter Notebook

# ------------------------ Configuration ------------------------

# Load spaCy Greek language model
try:
    nlp = spacy.load('el_core_news_sm')
except OSError:
    print("Downloading 'el_core_news_sm' model for spaCy as it was not found.")
    from spacy.cli import download
    download('el_core_news_sm')
    nlp = spacy.load('el_core_news_sm')

# Define the path to the folder containing the .txt files
folder_path = '/content'  # Replace with your folder path

# Define model parameters
max_tokens = 128  # Maximum number of tokens per chunk
overlap_sentences = 1  # Number of sentences to overlap between chunks

# Initialize the tokenizer associated with the model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Qdrant Configuration
qdrant_host = 'localhost'  # Qdrant host
qdrant_port = 6333         # Qdrant port
qdrant_collection_name = 'text_clusters'  # Name of the collection

# ------------------------ Functions ------------------------

def split_text_into_chunks(text, max_tokens=128, overlap_sentences=1):
    """
    Splits text into chunks based on sentences using spaCy, ensuring:
    - Each chunk starts and ends with complete sentences.
    - Chunks do not exceed max_tokens.
    - Overlaps are based on complete sentences.
    - Handles leading and trailing punctuation and whitespace.

    Parameters:
    - text (str): The input text to split.
    - max_tokens (int): Maximum number of tokens per chunk.
    - overlap_sentences (int): Number of sentences to overlap between chunks.

    Returns:
    - chunks (list): A list of text chunks.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_token_count = len(sentence_tokens)

        if current_chunk_tokens + sentence_token_count <= max_tokens:
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_token_count
            i += 1
        else:
            if current_chunk:
                # Join sentences to form the chunk
                chunk_text = ' '.join(current_chunk).strip()
                # Remove any leading or trailing periods and whitespace
                chunk_text = chunk_text.strip('.').strip()
                chunks.append(chunk_text)
                # Start new chunk with overlap
                if overlap_sentences > 0:
                    overlap = current_chunk[-overlap_sentences:]
                    # Clean overlap sentences
                    overlap = [s.strip() for s in overlap]
                    current_chunk = overlap.copy()
                    current_chunk_tokens = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk)
                else:
                    current_chunk = []
                    current_chunk_tokens = 0
            else:
                # Sentence itself exceeds max_tokens
                print(f"Warning: A single sentence exceeds max_tokens limit: {sentence[:50]}...")
                # Option 1: Split the long sentence further (e.g., by punctuation)
                sub_sentences = split_long_sentence(sentence, max_tokens)
                chunks.extend(sub_sentences)
                i += 1
                current_chunk = []
                current_chunk_tokens = 0

    # Add any remaining sentences as the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk).strip()
        chunk_text = chunk_text.strip('.').strip()
        chunks.append(chunk_text)

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
    sub_chunks = []
    punctuation = [".", ",", ";", ":", "-", "–", "—", "(", ")", "\"", "'"]
    sub_sentences = []
    current_sub = ""
    for char in sentence:
        current_sub += char
        if char in punctuation:
            sub_sentences.append(current_sub.strip())
            current_sub = ""
    if current_sub:
        sub_sentences.append(current_sub.strip())

    # Further split if necessary
    for sub in sub_sentences:
        tokens = tokenizer.encode(sub, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            sub_chunks.append(sub)
        else:
            # As a last resort, split by whitespace
            words = sub.split()
            temp = ""
            for word in words:
                word_tokens = tokenizer.encode(word, add_special_tokens=False)
                if len(tokenizer.encode(temp + " " + word, add_special_tokens=False)) <= max_tokens:
                    temp = temp + " " + word if temp else word
                else:
                    if temp:
                        sub_chunks.append(temp)
                    temp = word
            if temp:
                sub_chunks.append(temp)
    return sub_chunks

def read_and_split_documents(folder_path):
    """
    Reads all .txt files from the specified folder and splits them into chunks.
    """
    documents = {}
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    print(f"Found {len(txt_files)} .txt files: {txt_files}")
    if len(txt_files) == 0:
        print("No .txt files found in the specified folder.")
    for file_path in txt_files:
        print(f"\nProcessing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                if not text.strip():
                    print(f"Warning: The file {file_path} is empty.")
                else:
                    # Split the text into chunks
                    chunks = split_text_into_chunks(text, max_tokens, overlap_sentences)
                    print(f"Number of chunks created from {file_path}: {len(chunks)}")
                    # Store the chunks with metadata
                    documents[file_path] = chunks
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    return documents

def initialize_qdrant(client, collection_name, vector_size, distance_metric="Cosine"):
    """
    Initializes a Qdrant collection. Creates it if it doesn't exist.

    Parameters:
    - client (QdrantClient): The Qdrant client instance.
    - collection_name (str): Name of the collection.
    - vector_size (int): Dimension of the vector embeddings.
    - distance_metric (str): Distance metric to use ('Cosine', 'Euclidean', etc.).

    Returns:
    - None
    """
    existing_collections = client.get_collections().collections
    if collection_name not in [col.name for col in existing_collections]:
        print(f"Creating Qdrant collection '{collection_name}' with vector size {vector_size} and distance metric '{distance_metric}'.")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=distance_metric
            )
        )
    else:
        print(f"Qdrant collection '{collection_name}' already exists.")

def upsert_embeddings_qdrant(client, collection_name, embeddings, metadata, payload_fields):
    """
    Upserts embeddings and their metadata into a Qdrant collection.

    Parameters:
    - client (QdrantClient): The Qdrant client instance.
    - collection_name (str): Name of the collection.
    - embeddings (list or np.ndarray): List or array of embedding vectors.
    - metadata (list of dict): List of metadata dictionaries corresponding to each embedding.
    - payload_fields (list): List of metadata fields to store in Qdrant.

    Returns:
    - None
    """
    points = []
    for idx, (embedding, meta) in enumerate(zip(embeddings, metadata)):
        payload = {field: meta[field] for field in payload_fields}
        points.append(qmodels.PointStruct(id=idx, vector=embedding, payload=payload))
    client.upsert(collection_name=collection_name, points=points)
    print(f"Upserted {len(points)} points into Qdrant collection '{collection_name}'.")

def perform_similarity_search(client, collection_name, query_embedding, top_k=5):
    """
    Performs a similarity search in Qdrant and retrieves the top_k most similar passages.

    Parameters:
    - client (QdrantClient): The Qdrant client instance.
    - collection_name (str): Name of the collection.
    - query_embedding (list or np.ndarray): Embedding vector for the query.
    - top_k (int): Number of top similar passages to retrieve.

    Returns:
    - results (list): List of dictionaries containing similar passages and their metadata.
    """
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True
    )
    results = []
    for hit in search_result:
        result = {
            'score': hit.score,
            'passage': hit.payload.get('passage'),
            'document': hit.payload.get('document'),
            'passage_index': hit.payload.get('passage_index'),
            'cluster': hit.payload.get('cluster')
        }
        results.append(result)
    return results

# ------------------------ Main Pipeline ------------------------

def main():
    # Read and split the documents
    documents = read_and_split_documents(folder_path)

    # Flatten the documents into a list of passages and collect metadata
    passages = []
    metadata = []
    for doc_name, chunks in documents.items():
        for idx, chunk in enumerate(chunks):
            passages.append(chunk)
            metadata.append({
                'document': os.path.basename(doc_name),
                'passage_index': idx
            })

    print(f"\nTotal number of passages: {len(passages)}")

    if len(passages) == 0:
        print("No passages were extracted. Please check the document reading and splitting process.")
        return

    # Encode the passages into embeddings
    print("\nEncoding passages into embeddings...")
    embeddings = model.encode(passages, convert_to_tensor=False, show_progress_bar=True)

    # Optionally apply dimensionality reduction before clustering
    apply_pca = True  # Set to False if you do not want to apply PCA
    if apply_pca:
        print("\nApplying PCA to reduce dimensionality...")
        pca = PCA(n_components=50, random_state=42)
        reduced_embeddings = pca.fit_transform(embeddings)
        clustering_data = reduced_embeddings
    else:
        clustering_data = embeddings

    # Determine the optimal number of clusters using the Elbow Method
    print("\nDetermining the optimal number of clusters using the Elbow Method...")
    distortions = []
    K = range(2, min(11, len(passages) // 2 + 1))  # Fixed upper limit to avoid high K

    for k in K:
        kmeans_model = KMeans(n_clusters=k, random_state=42)
        kmeans_model.fit(clustering_data)
        distortions.append(kmeans_model.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 4))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.show()

    # Use KneeLocator to find the elbow point
    knee = KneeLocator(K, distortions, curve='convex', direction='decreasing')
    optimal_clusters_elbow = knee.knee

    if optimal_clusters_elbow is None:
        print("Elbow point not detected automatically.")
        optimal_clusters_elbow = int(input("Enter the optimal number of clusters based on the Elbow Method plot: "))
    else:
        print(f"Optimal number of clusters detected by Elbow Method: {optimal_clusters_elbow}")

    # Calculate silhouette scores for the same range of K
    print("\nCalculating silhouette scores for different values of K...")
    silhouette_scores = []
    for k in K:
        kmeans_model = KMeans(n_clusters=k, random_state=42)
        labels = kmeans_model.fit_predict(clustering_data)
        score = silhouette_score(clustering_data, labels)
        silhouette_scores.append(score)

    # Plot silhouette scores
    plt.figure(figsize=(8, 4))
    plt.plot(K, silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different k')
    plt.show()

    # Determine the best K based on silhouette score
    best_k_silhouette = K[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters based on silhouette score: {best_k_silhouette}")

    # Combine both methods to decide final K
    print(f"\nElbow Method suggests {optimal_clusters_elbow} clusters.")
    print(f"Silhouette Score suggests {best_k_silhouette} clusters.")

    if optimal_clusters_elbow == best_k_silhouette:
        final_k = optimal_clusters_elbow
        print(f"Both methods agree on {final_k} clusters.")
    else:
        # Choose based on higher silhouette score
        final_k = best_k_silhouette
        print(f"Choosing {final_k} clusters based on higher silhouette score.")

    # Perform KMeans clustering with the final number of clusters
    print(f"\nClustering embeddings into {final_k} clusters...")
    clustering_model = KMeans(n_clusters=final_k, random_state=42)
    clustering_model.fit(clustering_data)
    cluster_assignments = clustering_model.labels_

    # Update metadata with cluster assignments
    for idx, label in enumerate(cluster_assignments):
        metadata[idx]['cluster'] = int(label)

    # Map clusters to passages
    clusters = {}
    for idx, label in enumerate(cluster_assignments):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({
            'passage': passages[idx],
            'document': metadata[idx]['document'],
            'passage_index': metadata[idx]['passage_index']
        })

    # Write clusters to a CSV file
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

    # ------------------------ Qdrant Integration ------------------------

    # Initialize Qdrant client
    print("\nInitializing Qdrant client...")
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Determine the vector size from embeddings
    vector_size = len(embeddings[0])

    # Initialize Qdrant collection
    initialize_qdrant(client, qdrant_collection_name, vector_size, distance_metric="Cosine")

    # Prepare payload fields
    payload_fields = ['passage', 'document', 'passage_index', 'cluster']

    # Combine embeddings and metadata
    combined_metadata = metadata  # Each entry has 'document', 'passage_index', 'cluster'

    # Upsert embeddings into Qdrant
    print("\nUpserting embeddings and metadata into Qdrant...")
    upsert_embeddings_qdrant(
        client=client,
        collection_name=qdrant_collection_name,
        embeddings=embeddings,
        metadata=combined_metadata,
        payload_fields=payload_fields
    )

    # ------------------------ Visualization ------------------------

    # Optional: Visualize the clusters using t-SNE
    try:
        print("\nGenerating t-SNE visualization...")
        num_samples = len(embeddings)

        if num_samples < 3:
            print("Not enough samples for t-SNE visualization.")
        else:
            perplexity = min(30, num_samples - 1)
            print(f"Using perplexity={perplexity}")

            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
            embeddings_2d = tsne.fit_transform(clustering_data)

            # Create a DataFrame for plotting
            df = pd.DataFrame()
            df['x'] = embeddings_2d[:, 0]
            df['y'] = embeddings_2d[:, 1]
            df['Cluster'] = cluster_assignments.astype(str)
            df['Document'] = [m['document'] for m in metadata]

            # Plot the clusters
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=df, x='x', y='y', hue='Cluster', style='Document', palette='tab10', s=100)
            plt.title('t-SNE Visualization of Clusters')
            plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
    except ImportError:
        print("t-SNE visualization requires scikit-learn. Visualization skipped.")
    except ValueError as e:
        print(f"Error during t-SNE visualization: {e}")
        print("Adjust the perplexity parameter to be less than the number of samples.")

    # ------------------------ Similarity Search Example ------------------------

    # Example similarity search
    print("\nPerforming an example similarity search...")
    example_query = "Εισαγωγικό κείμενο για αναζήτηση."  # Replace with your query in Greek
    query_embedding = model.encode([example_query], convert_to_tensor=False)[0]

    similar_passages = perform_similarity_search(
        client=client,
        collection_name=qdrant_collection_name,
        query_embedding=query_embedding,
        top_k=5
    )

    print("\nTop 5 similar passages:")
    for idx, passage in enumerate(similar_passages, 1):
        print(f"\nRank {idx}:")
        print(f"Score: {passage['score']:.4f}")
        print(f"Document: {passage['document']}")
        print(f"Passage Index: {passage['passage_index']}")
        print(f"Passage: {passage['passage']}")

# ------------------------ Execute Pipeline ------------------------

if __name__ == "__main__":
    main()
