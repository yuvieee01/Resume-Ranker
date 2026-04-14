***

# Resume Ranker 📄🚀

An automated Resume Screening tool that ranks candidate resumes against a specific Job Description (JD) using Natural Language Processing (NLP) and Machine Learning techniques.

## 📌 Overview
This project simplifies the recruitment process by quantifying how well a resume matches a job description. It extracts text from multiple file formats, preprocesses the language to remove "noise," and uses **TF-IDF Vectorization** and **Cosine Similarity** to calculate a matching score.

## ✨ Features
- **Multi-Format Support**: Automatically extracts text from `.pdf`, `.docx`, and `.txt` files.
- **NLP Preprocessing**: Uses `spaCy` for tokenization, lemmatization, and removal of stop words/punctuation.
- **Advanced Vectorization**: Implements `TfidfVectorizer` with bigrams (ngram_range 1, 2) to capture context.
- **Similarity Scoring**: Ranks resumes based on **Cosine Similarity** (0 to 1 scale).
- **Data Visualization**: Generates a professional bar chart of the top 10 rankings using `Seaborn`.
- **Auto-Setup**: Automatically creates a sample `resumes/` directory with dummy data if none exists.

## 🛠️ Prerequisites
Before running the script, ensure you have Python installed. You will also need to download the English language model for `spaCy`.

```bash
# Required libraries
pip install pandas spacy scikit-learn matplotlib seaborn pdfminer.six docx2txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

## 🚀 How to Use
1. **Clone or Download** this repository.
2. **Prepare Resumes**: Place all candidate resumes in a folder named `resumes` in the same directory as the script.
3. **Run the Script**:
   ```bash
   python resume_ranker.py
   ```
4. **Input JD**: When prompted, paste the Job Description into the terminal.
5. **View Results**: The script will print a ranked table in the terminal and open a visualization window showing the top candidates.

## 🧠 How it Works
1. **Text Extraction**: The tool uses `pdfminer` and `docx2txt` to convert binary document formats into raw strings.
2. **Text Cleaning**: 
   - Converts text to lowercase.
   - Removes stop words (e.g., "the", "is", "at").
   - Lemmatization: Reduces words to their base form (e.g., "running" becomes "run").
3. **Vectorization**: The Job Description and all Resumes are converted into a **TF-IDF** (Term Frequency-Inverse Document Frequency) matrix. This highlights unique keywords that define a specific resume.
4. **Ranking**: It calculates the "distance" between the JD vector and Resume vectors using **Cosine Similarity**. A score of `1.0` would be a perfect match.

## 📊 Sample Output
**Terminal:**
```text
--- Top Ranked Resumes ---
                          filename  similarity_score
0    resume_1_python_django_dev.txt          0.452312
1  resume_3_python_data_scientist.txt          0.321455
2    resume_4_junior_python_dev.txt          0.284410
3                resume_2_java_dev.txt          0.120012
```

**Visualization:**
The script generates a horizontal bar chart displaying the `similarity_score` for the top 10 candidates for quick visual comparison.

## 📂 Project Structure
```text
.
├── resume_ranker.py      # Main application code
├── resumes/              # Folder containing candidate resumes (PDF, DOCX, TXT)
└── README.md             # Project documentation
```

## 📝 License
This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

***

### 💡 Tips for better results:
- Ensure resumes are text-based (not scanned images/OCRs).
- The more detailed the Job Description, the more accurate the ranking will be.
