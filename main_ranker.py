import os # For file and directory operations
import pandas as pd # For data manipulation and DataFrame handling
import spacy # For NLP tasks
from sklearn.feature_extraction.text import TfidfVectorizer # For TF-IDF vectorization
from sklearn.metrics.pairwise import cosine_similarity # For cosine similarity calculation
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For enhanced visualizations

# Libraries for parsing files
from pdfminer.high_level import extract_text # For PDF text extraction
import docx2txt # For DOCX text extraction

# Loading spaCy model..
# 'en_core_web_sm' is a small English model, efficient for our purpose.
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    print("Please run this command in your terminal: python -m spacy download en_core_web_sm")
    # This import is here for the automated download, which is a fallback
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


def extract_text_from_file(file_path):
    """
    Extracts raw text from PDF, DOCX, and TXT files.
    
    Args:
        file_path (str): The path to the file.

    Returns:
        str: The extracted text from the file, or an empty string if format is not supported or file is empty.
    """
    text = ""
    try:
        if file_path.endswith('.pdf'):
            # Using pdfminer.six to extract text from PDF
            text = extract_text(file_path)
        elif file_path.endswith('.docx'):
            # Using docx2txt to extract text from DOCX
            text = docx2txt.process(file_path)
        elif file_path.endswith('.txt'):
            # Standard file reading for TXT files
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
    except Exception as e:
        print(f"Error reading file {os.path.basename(file_path)}: {e}")
        return ""
    return text


def preprocess_text(text):
    """
    Cleans and preprocesses text using spaCy.
    - Tokenization
    - Stop word removal
    - Lemmatization
    - Removal of punctuation and non-alphabetic characters
    
    Args:
        text (str): The raw text to process.

    Returns:
        str: A string of cleaned, lemmatized tokens.
    """
    if not text:
        return ""
        
    # Create a spaCy doc object
    doc = nlp(text)
    
    processed_tokens = []
    for token in doc:
        # Check if the token is not a stop word, not punctuation, and is alphabetic
        if not token.is_stop and not token.is_punct and token.is_alpha:
            # Lemmatize the token and convert to lower case
            processed_tokens.append(token.lemma_.lower())

    # Join the tokens back into a single string
    return " ".join(processed_tokens)


def rank_resumes(job_description_text, resume_files_path):
    """
    Ranks resumes in a directory against a job description.
    
    Args:
        job_description_text (str): The raw text of the job description.
        resume_files_path (str): The path to the directory containing resume files.

    Returns:
        pandas.DataFrame: A DataFrame with ranked resumes and their scores.
    """
    # 1. Preprocess the Job Description
    processed_jd = preprocess_text(job_description_text)
    
    resume_data = []
    
    # 2. Loop through resume files, extract text, and preprocess
    print(f"Reading resumes from: {resume_files_path}")
    for filename in os.listdir(resume_files_path):
        file_path = os.path.join(resume_files_path, filename)
        if os.path.isfile(file_path):
            print(f"Processing {filename}...")
            resume_text = extract_text_from_file(file_path)
            if resume_text:
                processed_resume = preprocess_text(resume_text)
                resume_data.append({
                    "filename": filename,
                    "processed_text": processed_resume
                })
            else:
                print(f"Could not extract text from {filename}.")

    if not resume_data:
        print("No resumes found or could be processed in the directory.")
        return pd.DataFrame()

    # Create a list of all processed texts (JD + resumes) for the vectorizer
    all_texts = [processed_jd] + [resume['processed_text'] for resume in resume_data]
    
    # 3. TF-IDF Vectorization
    # Initialize the TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2)) # Use unigrams and bigrams
    
    # Fit and transform the texts to create TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
    
    # 4. Cosine Similarity Calculation
    # The first vector (index 0) is the job description
    # Compare the JD vector against all resume vectors (from index 1 onwards)
    cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    
    # 5. Compile and Rank Results
    # Get the scores from the resulting matrix
    scores = cosine_scores[0]
    
    # Add scores to our resume data
    for i, resume in enumerate(resume_data):
        resume['similarity_score'] = scores[i]
        
    # Create a pandas DataFrame for easy sorting and display
    results_df = pd.DataFrame(resume_data)
    results_df = results_df.sort_values(by='similarity_score', ascending=False)
    results_df = results_df.reset_index(drop=True)
    
    return results_df[['filename', 'similarity_score']]


def visualize_results(df):
    """
    Creates a bar chart of the resume ranking scores.
    
    Args:
        df (pandas.DataFrame): The DataFrame with ranked results.
    """
    if df.empty:
        print("Cannot visualize empty results.")
        return
        
    plt.figure(figsize=(12, 8))
    sns.barplot(x='similarity_score', y='filename', data=df.head(10), palette='viridis')
    
    plt.title('Top 10 Resume Rankings', fontsize=16)
    plt.xlabel('Cosine Similarity Score', fontsize=12)
    plt.ylabel('Resume Filename', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim(0, 1) # Scores are between 0 and 1
    
    # Add score labels to the bars
    for index, value in enumerate(df.head(10)['similarity_score']):
        plt.text(value, index, f" {value:.2f}", va='center')
        
    plt.tight_layout()
    print("Displaying ranking chart... Close the chart window to exit.")
    plt.show()


if __name__ == '__main__':

    RESUME_DIRECTORY = 'resumes' #Folder where the resumes are stored.
    
    # Paste your job description here
    JOB_DESCRIPTION = input("Enter Job Description:\n")
    # ---------------------

    # --- Create dummy resumes for testing if the directory is empty ---
    if not os.path.exists(RESUME_DIRECTORY):
        print(f"Creating dummy '{RESUME_DIRECTORY}' directory with sample resumes.")
        os.makedirs(RESUME_DIRECTORY)
        # Create dummy resumes (Table 3.1 content)
        resumes_content = {
            "resume_1_python_django_dev.txt": "Alice. Experienced Python developer skilled in Django, Flask, REST APIs, and PostgreSQL. Deployed applications on AWS. 6 years experience.",
            
            "resume_2_java_dev.txt": "Bob. Senior Java Developer with expertise in Spring Boot and microservices. Limited knowledge of Python.",
            
            "resume_3_python_data_scientist.txt": "Charlie. Data Scientist with strong Python skills. Proficient in pandas, NumPy, and scikit-learn. Some experience with Flask for building model APIs. Familiar with SQL.",
            
            "resume_4_junior_python_dev.txt": "David. Junior Python developer with 1 year experience. Familiar with Python basics and Django. Eager to learn about cloud deployment and databases like MongoDB."
        }

        for filename, content in resumes_content.items():
            with open(os.path.join(RESUME_DIRECTORY, filename), 'w') as f:
                f.write(content)
        print("Dummy files created. You can now add your own PDF/DOCX/TXT resumes to this folder.")
    # -----------------------------------------------------------------
    
    
    # Run the ranking process
    ranked_resumes_df = rank_resumes(JOB_DESCRIPTION, RESUME_DIRECTORY)
    
    # Display the results
    if not ranked_resumes_df.empty:
        print("\n--- Top Ranked Resumes ---")
        print(ranked_resumes_df)
        print("\n")
        
        # Visualize the results
        visualize_results(ranked_resumes_df)
    else:
        print("No results to display.")