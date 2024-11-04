
# HR Policy Documents and Testing Questions

This repository contains several HR policy documents in Greek and a list of questions designed to test the performance of a Retrieval-Augmented Generation (RAG) based bot. The documents have overlapping content, which makes them ideal for testing the bot's ability to correctly identify and differentiate between similar passages.

## Questions for Testing

The following questions are designed to test the bot's ability to handle and correctly respond to queries with overlapping content from the provided HR policy documents:

1. **Πότε πρέπει να υποβάλω αίτηση για άδεια μητρότητας ή πατρότητας;**
2. **Ποιοι κανόνες ασφάλειας και υγείας πρέπει να τηρούνται στον εργασιακό χώρο;**
3. **Πώς αξιολογείται η απόδοσή μου και ποια είναι τα κριτήρια;**
4. **Ποιες ευκαιρίες εκπαίδευσης και ανάπτυξης παρέχει η εταιρεία;**
5. **Τι περιλαμβάνει η πολιτική εργασιακής ισορροπίας;**
6. **Ποια προγράμματα υποστήριξης προσφέρονται για την υγεία και την ευεξία των εργαζομένων;**

---

## Clustering Tool

The `clustering_tool.py` script is a comprehensive tool designed to process, cluster, and manage HR policy documents. It leverages advanced Natural Language Processing (NLP) techniques and integrates with the Qdrant vector database for efficient storage and retrieval of text embeddings. This tool enhances the ability to manage large sets of documents, identify similar passages, and perform similarity searches effectively.

### Features

1. **Document Processing and Chunking**
   - Reads multiple `.txt` HR policy documents.
   - Splits documents into manageable chunks based on sentences while ensuring context preservation.
   - Handles overlapping content to maintain continuity between chunks.
   
2. **Embedding Generation**
   - Utilizes Sentence Transformers to convert text passages into high-dimensional embeddings.
   - Supports multilingual models, specifically tailored for Greek text.

3. **Clustering**
   - Implements the Elbow Method and Silhouette Score to determine the optimal number of clusters.
   - Performs KMeans clustering on the generated embeddings.
   - Visualizes clusters using t-SNE for intuitive understanding.

4. **Vector Database Integration**
   - Stores embeddings and associated metadata in Qdrant for efficient retrieval.
   - Ensures data persistence, allowing reloading of data without reprocessing.

5. **Similarity Search**
   - Enables users to perform similarity searches to find passages similar to a given query.
   - Retrieves relevant passages along with their source documents and cluster information.

## Installation

### Prerequisites

- **Python 3.7 or higher**: Ensure Python is installed on your system.
- **Docker**: Required for running the Qdrant vector database.

### Clone the Repository

```bash
git clone https://github.com/Argyx/HR_Policy_Assignment.git
cd hr-policy-clustering
```

### Install Python Dependencies

It's recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download el_core_news_sm
```

**If a `requirements.txt` is not working, install the necessary libraries manually:**

```bash
pip install spacy sentence-transformers scikit-learn transformers kneed pandas matplotlib seaborn tqdm qdrant-client
python -m spacy download el_core_news_sm
```

### Set Up Qdrant

Run Qdrant using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

To ensure data persistence across restarts, use Docker volumes:

```bash
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

## Usage

### Prepare Your Documents

Place all your `.txt` HR policy documents in a designated folder. Update the `folder_path` variable in `clustering_tool.py` to point to this folder.

```python
folder_path = '/src' 
```

### Run the Clustering Tool

Execute the script to process documents, generate embeddings, perform clustering, and store data in Qdrant.

```bash
python clustering_tool.py
```

### Script Workflow

1. **Document Processing**: Reads and splits documents into chunks.
2. **Embedding Generation**: Converts text chunks into embeddings.
3. **Clustering**: Determines optimal clusters and assigns passages to clusters.
4. **Qdrant Integration**: Stores embeddings and metadata in the Qdrant database.
5. **Visualization**: Generates a t-SNE plot of the clusters.
6. **Similarity Search**: Performs an example similarity search based on a sample query.

### Example Similarity Search

Modify the `example_query` variable within the script to test with different queries.

```python
example_query = "Εισαγωγικό κείμενο για αναζήτηση."  # Replace with your query in Greek
```

The script will output the top 5 similar passages along with their scores, source documents, and cluster information.

## Output

1. **CSV File**: `clusters.csv` containing all passages with their cluster assignments and metadata.
2. **Qdrant Collection**: Stores embeddings and associated metadata for efficient retrieval.
3. **Visualization**: A t-SNE plot visualizing the distribution of clusters.
4. **Similarity Search Results**: Displays top similar passages based on the provided query.

## Customization

### Adjusting Clustering Parameters

- **Maximum Tokens per Chunk**: Modify `max_tokens` to control the size of text chunks.

```python
max_tokens = 128  # Adjust based on requirements
```

- **Overlap Sentences**: Change `overlap_sentences` to control the number of overlapping sentences between chunks.

```python
overlap_sentences = 1  # Adjust as needed
```

- **Dimensionality Reduction**: Toggle PCA application by setting `apply_pca`.

```python
apply_pca = True  # Set to False to skip PCA
```

### Qdrant Configuration

- **Host and Port**: Update `qdrant_host` and `qdrant_port` if Qdrant is hosted elsewhere.

```python
qdrant_host = 'localhost'  # Replace if different
qdrant_port = 6333         # Replace if different
```

- **Collection Name**: Change `qdrant_collection_name` to customize the Qdrant collection.

```python
qdrant_collection_name = 'text_clusters'  # Customize as desired
```

## Troubleshooting

- **Qdrant Connection Issues**: Ensure Qdrant is running and accessible at the specified host and port.
- **Missing spaCy Model**: The script attempts to download the `el_core_news_sm` model if not found. Ensure you have internet connectivity.
- **Large Documents**: For extremely large documents, consider increasing `max_tokens` or enhancing the sentence splitting logic.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for any enhancements or bug fixes.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgements

- [spaCy](https://spacy.io/) for NLP processing.
- [Sentence Transformers](https://www.sbert.net/) for embedding generation.
- [Qdrant](https://qdrant.tech/) for vector database management.
- [Kneed](https://github.com/arvkevi/kneed) for elbow method detection.
- [Scikit-learn](https://scikit-learn.org/) for clustering and evaluation metrics.
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for data visualization.

---

## Additional Notes

- **Data Privacy**: Ensure that all HR policy documents comply with your organization's data privacy policies before processing and storing them.
- **Scalability**: The current setup is suitable for moderate-sized document collections. For larger datasets, consider optimizing the embedding generation and clustering steps or deploying Qdrant in a scalable environment.
- **Extensibility**: The `clustering_tool.py` script can be extended to support additional functionalities such as real-time clustering updates, integration with other databases, or enhanced NLP preprocessing steps.

Feel free to reach out for any questions or support regarding the usage of the clustering tool!
