
# HR Policy Clustering and Similarity Search Tool

This repository contains multiple HR policy documents in Greek and a list of questions designed to test the performance of a Retrieval-Augmented Generation (RAG)-based bot. The setup is built to process, cluster, and manage these documents while performing similarity searches, leveraging Qdrant as a vector database.

## Questions for Testing

The following questions help evaluate the bot's ability to handle overlapping content in HR policies:

1. **Πότε πρέπει να υποβάλω αίτηση για άδεια μητρότητας ή πατρότητας;**
2. **Ποιοι κανόνες ασφάλειας και υγείας πρέπει να τηρούνται στον εργασιακό χώρο;**
3. **Πώς αξιολογείται η απόδοσή μου και ποια είναι τα κριτήρια;**
4. **Ποιες ευκαιρίες εκπαίδευσης και ανάπτυξης παρέχει η εταιρεία;**
5. **Τι περιλαμβάνει η πολιτική εργασιακής ισορροπίας;**
6. **Ποια προγράμματα υποστήριξης προσφέρονται για την υγεία και την ευεξία των εργαζομένων;**

---

## Clustering Tool Overview

The `clustering_tool.py` script is designed to process, cluster, and manage HR policy documents. It uses advanced NLP techniques and integrates with the Qdrant vector database for efficient storage and retrieval of text embeddings.

### Features

1. **Document Processing and Chunking**
   - Reads `.txt` HR policy documents and splits them into chunks, preserving context and handling overlapping content.

2. **Embedding Generation**
   - Uses Sentence Transformers for high-dimensional embeddings, specifically tailored for Greek text.

3. **Clustering**
   - Determines the optimal number of clusters with the Elbow Method and performs KMeans clustering.
   - Visualizes clusters using t-SNE for intuitive understanding.

4. **Vector Database Integration**
   - Stores embeddings and metadata in Qdrant for retrieval and persistence.

5. **Similarity Search**
   - Allows similarity searches to find passages related to a given query.
   - Retrieves relevant passages with source documents and cluster info.

## Installation

### Prerequisites

- **Python 3.7+**: Ensure Python is installed on your system.
- **Docker**: Required for running the Qdrant vector database.

### Clone the Repository

```bash
git clone https://github.com/Argyx/HR_Policy_Assignment.git
cd HR_Policy_Assignment
```

### Install Python Dependencies

It's recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Set Up Qdrant

Run Qdrant using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

For data persistence:

```bash
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

## Usage

### Prepare Your Documents

Place all `.txt` HR policy documents in the folder specified in `folder_path` in `clustering_tool.py`.

### Run the Clustering Tool

Execute the script to process documents, generate embeddings, perform clustering, and store data in Qdrant.

```bash
python clustering_tool.py
```

### Script Workflow

1. **Document Processing**: Reads and chunks documents.
2. **Embedding Generation**: Converts text chunks to embeddings.
3. **Clustering**: Finds optimal clusters and assigns passages to clusters.
4. **Qdrant Integration**: Stores embeddings and metadata in Qdrant.
5. **Visualization**: Generates a t-SNE plot of clusters.
6. **Similarity Search**: Provides top similar passages based on sample queries.

### Example Similarity Search

Update the `example_query` in the script to try different queries.

## Output

1. **CSV File**: `clusters.csv` containing passages, clusters, and metadata.
2. **Qdrant Collection**: Stores embeddings and metadata.
3. **Visualization**: t-SNE plot of clusters.
4. **Similarity Search Results**: Displays top matches for a query.

## Customization

### Clustering Parameters

- **Maximum Tokens per Chunk**: Adjust `max_tokens` for text chunk size.

```python
max_tokens = 128
```

- **Qdrant Config**: Update `qdrant_host` and `qdrant_port` if hosted externally.

```python
qdrant_host = 'localhost'
qdrant_port = 6333
```

## Troubleshooting

- **Qdrant Issues**: Check Docker setup if connection errors occur.
- **Large Docs**: Adjust `max_tokens` for extremely large documents.

## Contributing

Contributions are welcome! Open issues or submit PRs for enhancements.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
