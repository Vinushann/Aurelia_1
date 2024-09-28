1. Fetch all remote branches:               git fetch origin
2. List all branches:                       git branch -r
3. Check out the new branch:                git checkout <branch name>
4. Pull the changes from the new branch:    git pull origin <branch name>

# **Simple Information Retrieval System**

Welcome to the **Simple Information Retrieval System** project! This application is designed to efficiently search and retrieve relevant PDF documents based on user queries. It leverages advanced natural language processing (NLP) techniques and machine learning algorithms to provide accurate and speedy search results.

---

## **Table of Contents**

- [Project Overview](#project-overview)
- [Features](#features)
- [Demo](#demo)
- [System Architecture](#system-architecture)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Set Up the Environment](#set-up-the-environment)
  - [Install Dependencies](#install-dependencies)
  - [Download NLTK Data](#download-nltk-data)
  - [Prepare the Data](#prepare-the-data)
  - [Run the Application](#run-the-application)
- [Usage](#usage)
  - [Starting the Application](#starting-the-application)
  - [Performing a Search](#performing-a-search)
  - [Using Keyboard Shortcuts](#using-keyboard-shortcuts)
  - [Opening and Downloading Documents](#opening-and-downloading-documents)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## **Project Overview**

The Simple Information Retrieval System is a Python-based application that allows users to search through a collection of PDF documents. It preprocesses documents using NLP techniques, builds an inverted index for efficient searching, and ranks documents using TF-IDF and cosine similarity measures.

The frontend is built using **Streamlit**, providing an interactive web interface where users can input queries, customize the number of results, and access retrieved documents. The system also supports keyboard shortcuts for enhanced usability.

---

## **Features**

- **Efficient Search and Retrieval**: Quickly search through a collection of PDF documents.
- **Natural Language Processing**: Utilizes lemmatization and stop word removal for better search accuracy.
- **Query Expansion**: Enhances user queries by adding synonyms using WordNet.
- **Document Ranking**: Ranks search results based on TF-IDF weighting and cosine similarity.
- **Interactive User Interface**: Built with Streamlit for a responsive and user-friendly experience.
- **Keyboard Shortcuts**:
  - Press `/` to focus on the search input field.
  - Press number keys `1` to `5` to open corresponding documents.
- **Open and Download Documents**: Access retrieved documents directly from the interface.

---

## **Demo**

**[Insert Screenshots or GIFs Here]**

- *Figure 1*: Search Interface
- *Figure 2*: Search Results
- *Figure 3*: Document Preview

---

## **System Architecture**

The system consists of two main components:

1. **Backend**:
   - Document Preprocessing
   - Indexing
   - Query Processing
   - Ranking

2. **Frontend**:
   - User Interface built with Streamlit
   - Interactive elements for user input and result display

**[Insert System Architecture Diagram Here]**

---

## **Installation**

### **Prerequisites**

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.8 or higher
- **Web Browser**: Latest version of Chrome, Firefox, or Edge

### **Clone the Repository**

```bash
git clone https://github.com/your-username/simple-information-retrieval-system.git
cd simple-information-retrieval-system
```

### **Set Up the Environment**

Create a virtual environment to manage dependencies.

```bash
python -m venv venv
# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### **Install Dependencies**

Install the required Python packages using `requirements.txt`.

```bash
pip install -r requirements.txt
```

**requirements.txt** includes:

```
streamlit
scikit-learn
nltk
PyPDF2
numpy
pandas
```

*Note: `pandas` is optional and only needed if used in your project.*

### **Download NLTK Data**

Download the necessary NLTK datasets.

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

Alternatively, run the script:

```bash
python download_nltk_data.py
```

*Create a `download_nltk_data.py` script with the above code if not existing.*

### **Prepare the Data**

1. **Place Your PDF Documents**:

   - Put all your PDF files into the `pdfs` directory within the project folder.

2. **Run Preprocessing Scripts**:

   - **Preprocess Documents**:

     ```bash
     python preprocess_documents.py
     ```

   - **Vectorize Documents**:

     ```bash
     python vectorization.py
     ```

   - These scripts will extract text from PDFs, preprocess the text, and build the TF-IDF matrix.

### **Run the Application**

1. **Start the HTTP Server to Serve PDFs**:

   ```bash
   python -m http.server 8502
   ```

2. **Run the Streamlit App**:

   ```bash
   streamlit run app.py
   ```

3. **Access the Application**:

   - Open your web browser and go to `http://localhost:8501`.

---

## **Usage**

### **Starting the Application**

Ensure that both the HTTP server and Streamlit app are running as per the installation instructions.

### **Performing a Search**

1. **Enter Your Query**:

   - Click on the search input field or press `/` to focus.
   - Type your search query (e.g., "machine learning algorithms").

2. **Select Number of Results**:

   - Choose how many results you want to display (1 to 10).

3. **Initiate Search**:

   - Click the **Search** button or press `Enter`.

### **Using Keyboard Shortcuts**

- **Focus Search Field**: Press `/`.
- **Open Documents**:
  - Press `1` to open the first document.
  - Press `2` to open the second document.
  - Continue up to `5`.

*Note: Keyboard shortcuts work only when the input field is not focused.*

### **Opening and Downloading Documents**

- **Open PDF**:

  - Click on the **Open PDF** link below a search result to view the document in a new browser tab.

- **Download PDF**:

  - Click the **Download PDF** button to download the document to your device.

---

## **Project Structure**

```
simple-information-retrieval-system/
├── app.py
├── preprocess_documents.py
├── vectorization.py
├── query_processing.py
├── download_nltk_data.py
├── requirements.txt
├── README.md
├── pdfs/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── data/
│   ├── preprocessed_documents.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   └── ...
├── images/
│   ├── interface.png
│   ├── results.png
│   └── ...
└── ...
```

- **app.py**: Main application script.
- **preprocess_documents.py**: Script for preprocessing PDF documents.
- **vectorization.py**: Script for vectorizing documents using TF-IDF.
- **query_processing.py**: Module containing query preprocessing functions.
- **download_nltk_data.py**: Script to download NLTK data.
- **requirements.txt**: List of project dependencies.
- **pdfs/**: Directory containing PDF documents.
- **data/**: Directory containing serialized data files.
- **images/**: Directory containing images for README and documentation.

---

## **Technologies Used**

- **Python 3.8+**
- **Streamlit**: Web application framework for the frontend.
- **NLTK (Natural Language Toolkit)**: For text preprocessing and NLP tasks.
- **scikit-learn**: For TF-IDF vectorization and cosine similarity calculation.
- **PyPDF2**: For extracting text from PDF documents.
- **NumPy**: For numerical computations.
- **Pandas** (optional): For data manipulation if used.
- **WordNet**: Lexical database for English, used for query expansion.

---

## **Contributing**

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**:

   - Click the **Fork** button at the top right of this page.

2. **Clone Your Fork**:

   ```bash
   git clone https://github.com/your-username/simple-information-retrieval-system.git
   ```

3. **Create a Branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**:

   - Add new features or fix bugs.

5. **Commit and Push**:

   ```bash
   git add .
   git commit -m "Add your message here"
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**:

   - Go to the original repository and create a pull request from your fork.

---

## **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## **Contact**

- **Your Name**
- **Email**: [your.email@example.com](mailto:your.email@example.com)
- **GitHub**: [your-username](https://github.com/your-username)

Feel free to reach out if you have any questions or need assistance with the project.

---

*Thank you for using the Simple Information Retrieval System!*

---

**Note**: Remember to replace placeholders like `your-username`, `your.email@example.com`, and any other personal details with your actual information. Also, ensure that any referenced images or files exist in the specified locations within your repository.
