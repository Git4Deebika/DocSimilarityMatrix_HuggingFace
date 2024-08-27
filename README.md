# Document Similarity Analysis
This Python script analyzes the similarity between multiple DOCX and RTF files using Hugging Face transformer models. It generates a similarity matrix and saves it to a CSV file.

## Installation

**Create a Virtual Environment (Recommended):**
   
   python -m venv my_env
   my_env\Scripts\activate`
     
**Install dependencies**
   pip install -r requirements.txt

**Run the script**
   python main.py

**Enter directory path:**
   The script will prompt you to enter the path to the directory containing your files.

**View output**
     The script will print the similarity matrix to the console and save it as a CSV file named similarity_matrix_YYYYMMDD_HHMMSS.csv (where YYYYMMDD_HHMMSS is the current timestamp).
