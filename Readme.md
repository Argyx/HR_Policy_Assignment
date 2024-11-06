
# HR Policy Clustering and Similarity Search Tool

This repository contains multiple HR policy documents in Greek, alongside a set of testing questions aimed at evaluating the performance of a Retrieval-Augmented Generation (RAG)-based bot. This tool processes, clusters, and manages HR policy documents, providing similarity search capabilities with Qdrant as a vector database.

## Questions for Testing

The following questions are designed to test the bot's ability to handle overlapping content from HR policies:

1. **Πότε πρέπει να υποβάλω αίτηση για άδεια μητρότητας ή πατρότητας;**
2. **Ποιοι κανόνες ασφάλειας και υγείας πρέπει να τηρούνται στον εργασιακό χώρο;**
3. **Πώς αξιολογείται η απόδοσή μου και ποια είναι τα κριτήρια;**
4. **Ποιες ευκαιρίες εκπαίδευσης και ανάπτυξης παρέχει η εταιρεία;**
5. **Τι περιλαμβάνει η πολιτική εργασιακής ισορροπίας;**
6. **Ποια προγράμματα υποστήριξης προσφέρονται για την υγεία και την ευεξία των εργαζομένων;**

---

## Overview of the Clustering Tool

The `clustering_tool.py` script is designed to process, cluster, and manage HR policy documents in Greek. It uses advanced NLP techniques and integrates with Qdrant for efficient storage and retrieval of text embeddings.

### Key Features

1. **Document Processing and Chunking**
   - Reads `.txt` HR policy documents and splits them into chunks, preserving context and managing overlapping content.
   
2. **Embedding Generation**
   - Uses Sentence Transformers to generate high-dimensional embeddings for Greek text.

3. **Clustering**
   - Determines optimal clusters via the Elbow Method and performs KMeans clustering.
   - Visualizes clusters with t-SNE for easy interpretation.

4. **Vector Database Integration**
   - Stores embeddings and metadata in Qdrant for easy retrieval and persistence.

5. **Similarity Search**
   - Provides similarity search to find related passages for given queries, returning relevant passages with document and cluster information.

---

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

Use a virtual environment for isolation:

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

---

## Usage

### Document Preparation

Place all `.txt` HR policy documents in the folder specified in `folder_path` within `clustering_tool.py`.

### Running the Tool

Run the script to process documents, generate embeddings, perform clustering, and store data in Qdrant.

```bash
python clustering_tool.py
```

---

## Script Workflow

1. **Document Processing**: Reads and splits documents into manageable chunks while preserving context.
2. **Embedding Generation**: Converts text chunks into embeddings.
3. **Clustering**: Determines optimal clusters and assigns passages.
4. **Qdrant Integration**: Stores embeddings and metadata for efficient retrieval.
5. **Visualization**: Creates a t-SNE plot to visualize clusters.
6. **Similarity Search**: Executes sample similarity searches for each query.

---

## Outputs

### 1. `clusters.csv`

This CSV file contains all passages, cluster assignments, and metadata:

- **Cluster**: The assigned cluster ID.
- **Document**: The document name.
- **Passage Index**: The passage's index in the document.
- **Passage**: The text content of the passage.

### 2. Qdrant Collection

The embeddings and associated metadata are stored in a Qdrant collection. This enables efficient similarity search queries and data persistence.

### 3. Visualization: `tsne_clusters.png`

A t-SNE plot is generated to visualize the distribution of clusters. Each point represents a passage, colored by cluster assignment.

### 4. Similarity Report: `similarity_report.html`

This HTML report is generated for each query, showing the top 3 most similar passages found. Each passage is highlighted to indicate similar content words, based on a configurable similarity threshold.

#### Structure of `similarity_report.html`

Each query section includes:

- **Query**: The query text displayed in bold.
- **Similar Passages**: The top 3 similar passages are shown in an ordered list, each including:
  - **Score**: Similarity score between the query and the passage.
  - **Document**: The name of the document containing the passage.
  - **Passage Index**: Index of the passage within the document.
  - **Highlighted Passage**: The passage text with similar words marked using HTML `<mark>` tags.

This structure is helpful for visual inspection of the model's performance and allows easy testing of various queries against the stored document embeddings.

---

## Customization

### Adjusting Clustering Parameters

- **Maximum Tokens per Chunk**: Modify `max_tokens` in `clustering_tool.py` to control text chunk size.

```python
max_tokens = 128
```

- **Qdrant Configuration**: Update `qdrant_host` and `qdrant_port` if Qdrant is hosted externally.

```python
qdrant_host = 'localhost'
qdrant_port = 6333
```

---

## Troubleshooting

- **Qdrant Issues**: Verify Docker setup if connection errors occur.
- **Large Documents**: Adjust `max_tokens` or optimize sentence splitting for large files.

---

## Contributing

Contributions are welcome! Open issues or submit pull requests for enhancements.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## Acknowledgements

- [Sentence Transformers](https://www.sbert.net/) for embedding generation.
- [Qdrant](https://qdrant.tech/) for vector database management.
- [Kneed](https://github.com/arvkevi/kneed) for elbow method detection.
- [Scikit-learn](https://scikit-learn.org/) for clustering and evaluation metrics.
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for data visualization.
git 