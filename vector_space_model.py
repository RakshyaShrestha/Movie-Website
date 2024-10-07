import os
import math
import numpy as np

# Function to tokenize the text
def tokenize(text):
    return text.lower().split()

# Function to calculate term frequency (TF)
def term_frequency(term, document):
    return document.count(term) / len(document)

# Function to calculate inverse document frequency (IDF)
def inverse_document_frequency(term, all_documents):
    num_docs_containing_term = sum(1 for doc in all_documents if term in doc)
    return math.log(len(all_documents) / (1 + num_docs_containing_term))

# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Function to compute TF-IDF for a document
def compute_tfidf(document, all_documents, vocab):
    tfidf_vector = []
    for term in vocab:
        tf = term_frequency(term, document)
        idf = inverse_document_frequency(term, all_documents)
        tfidf_vector.append(tf * idf)
    return np.array(tfidf_vector)

# Main function
def main():
    # Directory containing the text documents
    directory = 'Datasets'  # or provide the full path here

    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return  # Exit the function if the directory is not found

    # Reading all files from the directory
    docs = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r") as file:
                content = file.read()
                docs.append(content)
                filenames.append(filename)

    # Hardcoded queries
    queries = ['drama',
               'animation',
               'horror']

    # Tokenizing documents and queries
    tokenized_docs = [tokenize(doc) for doc in docs]
    tokenized_queries = [tokenize(query) for query in queries]

    # Building the vocabulary (unique words across all documents)
    vocab = sorted(set([word for doc in tokenized_docs for word in doc]))

    # Calculate TF-IDF vectors for documents and queries
    doc_tfidf_vectors = [compute_tfidf(doc, tokenized_docs, vocab) for doc in tokenized_docs]
    query_tfidf_vectors = [compute_tfidf(query, tokenized_docs, vocab) for query in tokenized_queries]

    # Calculate cosine similarities
    cosine_similarities = []
    for query_vector in query_tfidf_vectors:
        similarities = [cosine_similarity(query_vector, doc_vector) for doc_vector in doc_tfidf_vectors]
        cosine_similarities.append(similarities)

    # Write the ranked results to a text file
    with open("cosine_similarities_ranked_results.txt", "w") as output_file:
        for i, query in enumerate(queries):
            output_file.write(f"\nRanked cosine similarities for query '{query}':\n")

            # Pair document filenames with their similarity scores
            doc_similarity_pairs = list(zip(filenames, cosine_similarities[i]))
            # Sort by similarity in descending order
            ranked_docs = sorted(doc_similarity_pairs, key=lambda x: x[1], reverse=True)

            # Write ranked results
            for rank, (filename, score) in enumerate(ranked_docs, 1):
                output_file.write(f"Rank {rank}: Document {filename} - Score: {score:.4f}\n")

    # Optional: print ranked results for checking
    for i, query in enumerate(queries):
        print(f"\nRanked cosine similarities for query '{query}':")

        # Pair document filenames with their similarity scores
        doc_similarity_pairs = list(zip(filenames, cosine_similarities[i]))
        # Sort by similarity in descending order
        ranked_docs = sorted(doc_similarity_pairs, key=lambda x: x[1], reverse=True)

        # Print ranked results
        for rank, (filename, score) in enumerate(ranked_docs, 1):
            print(f"Rank {rank}: Document {filename} - Score: {score:.4f}")

if __name__ == "__main__":
    main()
