import os
import nltk
import glob
import string
import pickle
import PyPDF2
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pdfminer.high_level import extract_text

# Ensure necessary NLTK data files are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def extract_text_from_pdfs(pdf_folder_path):
    pdf_files = glob.glob(os.path.join(pdf_folder_path, '*.pdf'))
    documents = {}
    for pdf_file in pdf_files:
        text = extract_text(pdf_file)
        doc_id = os.path.basename(pdf_file)
        if not text.strip():
            print(f"Document {doc_id} is empty after text extraction.")
        else:
            print(f"Document {doc_id} has content after text extraction.")
        documents[doc_id] = text
    return documents

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation (keep numbers)
    punctuations = string.punctuation.replace('-', '')  # Keep hyphens (if needed)
    text = text.translate(str.maketrans('', '', punctuations))
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back to string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def preprocess_documents(documents):
    preprocessed_docs = {}
    for doc_id, text in documents.items():
        preprocessed_text = preprocess_text(text)
        if not preprocessed_text.strip():
            print(f"Document {doc_id} is empty after preprocessing.")
        else:
            print(f"Document {doc_id} has content after preprocessing.")
        preprocessed_docs[doc_id] = preprocessed_text
    return preprocessed_docs


if __name__ == '__main__':
    pdf_folder_path = './Doc/Raw'  # Update this path
    documents = extract_text_from_pdfs(pdf_folder_path)
    preprocessed_documents = preprocess_documents(documents)
    # Save preprocessed documents for later use
    with open('preprocessed_documents.pkl', 'wb') as f:
        pickle.dump(preprocessed_documents, f)