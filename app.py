import os
import pickle
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics.pairwise import cosine_similarity
from query_processing import preprocess_query, query_expansion

# Load the vectorizer and TF-IDF matrix
@st.cache_resource
def load_vectorizer_and_matrix():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix, doc_ids = pickle.load(f)
    return vectorizer, tfidf_matrix, doc_ids

vectorizer, tfidf_matrix, doc_ids = load_vectorizer_and_matrix()

# Load document snippets
@st.cache_resource
def load_document_snippets():
    with open('doc_snippets.pkl', 'rb') as f:
        doc_snippets = pickle.load(f)
    return doc_snippets

doc_snippets = load_document_snippets()

# Set the path to your PDF documents
pdf_folder_path = './static/pdfs'  

# Set the base URL for PDFs served by the HTTP server
pdf_base_url = 'http://localhost:8502/static/pdfs' 

def main():
    if 'results' not in st.session_state:
        st.session_state['results'] = None

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

        # Store the results in st.session_state
        num_results = 10  # You may need to define this based on how many results you want to show
        st.session_state['results'] = {
            'ranked_docs': ranked_docs,
            'num_results': num_results
        }

        display_results = True
    else:
        st.warning("Please enter a query.")
        display_results = False

    if st.session_state['results'] is not None:
        ranked_docs = st.session_state['results']['ranked_docs']
        num_results = st.session_state['results']['num_results']
        display_results = True
    else:
        display_results = False

    # Display results if available
    if display_results:
        st.header("Search Results")
        results_found = False

        for idx, (doc_id, score) in enumerate(ranked_docs[:num_results]):
            if score > 0:
                results_found = True
                doc_path = os.path.join(pdf_folder_path, doc_id)
                pdf_url = f"{pdf_base_url}/{doc_id}"

                st.write(f"**Document {idx + 1}:** {doc_id}")
                st.write(f"**Relevance Score:** {score:.4f}")

                # Display snippet
                snippet = doc_snippets.get(doc_id, "")
                st.write(snippet)

                # Create two columns for buttons
                button_col1, button_col2 = st.columns([1, 1])

                with button_col1:
                    # Open PDF button styled with HTML
                    st.markdown(
                        f'<a href="{pdf_url}" target="_blank" class="pdf-link"><button style="width:100%">Open PDF</button></a>',
                        unsafe_allow_html=True
                    )

                with button_col2:
                    # Download PDF button
                    with open(doc_path, 'rb') as f:
                        pdf_data = f.read()
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name=doc_id,
                        mime='application/pdf'
                    )

                # Add divider
                st.markdown('---')
            else:
                break
        if not results_found:
            st.write("No relevant documents found.")

if __name__ == '__main__':
    main()
