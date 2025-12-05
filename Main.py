import os
import zipfile
import datetime

import docx2txt
import rtf_converter  # Import library for RTF handling
import PyPDF2 # New library for PDF handling

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Disable HF Hub warnings if not needed
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Get the directory path from user input (or hardcoded)
# doc_directory = input("Enter the directory path containing your DOCX, RTF, and PDF files: ")
doc_directory = "C:/Users/2065276/Downloads/SampleWordFiles/Word_Files"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or "" # Use 'or ""' to handle None return
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None
    return text

# Check if the provided path is a valid directory
if not os.path.isdir(doc_directory):
    print("Error: The provided path is not a valid directory.")
else:
    # 1. Update File Listing to include DOCX, RTF, and PDF files
    print("Scanning directory for .docx, .rtf, and .pdf files...")
    allowed_extensions = ('.docx', '.rtf', '.RTF', '.pdf', '.PDF')
    doc_files = [os.path.join(doc_directory, f)
                 for f in os.listdir(doc_directory)
                 if f.endswith(allowed_extensions)]

    # 2. Document Processing & Conversion
    processed_texts = []
    file_paths_for_embeddings = [] # Only include files that were successfully processed
    for file_path in doc_files:
        try:
            text = ""
            if file_path.endswith('.docx'):
                text = docx2txt.process(file_path)
            elif file_path.endswith('.rtf') or file_path.endswith('.RTF'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    rtf_content = file.read()
                    text = rtf_converter.rtf_to_txt(rtf_content)
            elif file_path.endswith('.pdf') or file_path.endswith('.PDF'): # New PDF Logic
                text = extract_text_from_pdf(file_path)
            
            if text is not None and text.strip(): # Only add if text extraction was successful and not empty
                processed_texts.append(text)
                file_paths_for_embeddings.append(file_path)
            else:
                 print(f"Skipping {os.path.basename(file_path)}: Empty or failed text extraction.")

        except (zipfile.BadZipFile, OSError, UnicodeDecodeError, NotImplementedError) as e:
            print(f"Error processing {file_path}: {e}. Skipping...")


    # 3. Initialize Hugging Face Transformer Model and Tokenizer
    model_name = "w601sxs/b1ade-embed"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # 4. Generate Embeddings
    embeddings = []
    for doc_text in processed_texts:
        inputs = tokenizer(doc_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        embeddings.append(last_hidden_hidden_state.mean(dim=1).numpy().squeeze())

    if not embeddings:
        print("Error: No valid embeddings were generated. Please check your files.")
    else:
        # 5. Calculate Similarity Matrix
        similarity_matrix = cosine_similarity(embeddings)

        # 6. Create DataFrame
        filenames = [os.path.basename(file_path) for file_path in file_paths_for_embeddings] # Use the successful file paths
        df_similarity = pd.DataFrame(similarity_matrix, index=filenames, columns=filenames)

        # Display or save the similarity matrix
        print("\n--- Similarity Matrix ---")
        print(df_similarity)

        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        df_similarity.to_csv(f'similarity_matrix_{timestamp}.csv')
        print(f"\nSaved similarity matrix to similarity_matrix_{timestamp}.csv")
