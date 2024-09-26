import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from query_processing import preprocess_query, query_expansion
import os

# Load the vectorizer and TF-IDF matrix
@st.cache_resource
def load_vectorizer_and_matrix():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix, doc_ids = pickle.load(f)
    return vectorizer, tfidf_matrix, doc_ids

vectorizer, tfidf_matrix, doc_ids = load_vectorizer_and_matrix()

# Set the path to your PDF documents
pdf_folder_path = './Doc/Raw'  # Update this path

# Main Streamlit app
def main():
    st.title("Document Search Engine")

    # Input query from user
    query = st.text_input("Enter your search query:")

    if query:
        # Process the query
        preprocessed_query = preprocess_query(query)
        tokens = preprocessed_query.split()
        expanded_tokens = query_expansion(tokens)
        expanded_query = ' '.join(expanded_tokens)

        # Transform the query using the vectorizer
        query_vector = vectorizer.transform([expanded_query])

        # Compute similarity scores
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]

        # Rank documents
        doc_scores = list(zip(doc_ids, similarity_scores))
        ranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        # Display the results
        st.header("Search Results")
        results_found = False

        for doc_id, score in ranked_docs[:10]:
            if score > 0:
                results_found = True
                doc_path = os.path.join(pdf_folder_path, doc_id)
                # Create a link to open the PDF in a new tab
                with open(doc_path, 'rb') as f:
                    pdf_data = f.read()
                st.write(f"**Document:** {doc_id}")
                st.write(f"**Relevance Score:** {score:.4f}")
                st.download_button(
                    label="Open PDF",
                    data=pdf_data,
                    file_name=doc_id,
                    mime='application/pdf'
                )
            else:
                break
        if not results_found:
            st.write("No relevant documents found.")

if __name__ == '__main__':
    main()