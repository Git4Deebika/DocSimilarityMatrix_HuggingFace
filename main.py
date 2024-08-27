import os
import zipfile
import datetime

import docx2txt
import rtf_converter  # Import library for RTF handling

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel  # , AutoModelForMaskedLM
import torch

# Disable HF Hub warnings if not needed
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Get the directory path from user input
# doc_directory = input("Enter the directory path containing your DOCX and RTF files: ")
doc_directory = "C:/Users/2065276/Downloads/SampleWordFiles/Word_Files"

# Check if the provided path is a valid directory
if not os.path.isdir(doc_directory):
    print("Error: The provided path is not a valid directory.")
else:
    # Get a list of all DOCX and RTF files in the directory
    doc_files = [os.path.join(doc_directory, f)
                 for f in os.listdir(doc_directory)
                 if f.endswith('.docx') or f.endswith('.rtf') or f.endswith('.RTF')]

    # 1. Document Processing & Conversion
    processed_texts = []
    for file_path in doc_files:
        try:
            text = ""
            if file_path.endswith('.docx'):
                text = docx2txt.process(file_path)
            elif file_path.endswith('.rtf') or file_path.endswith('.RTF'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:  # Handle encoding issues
                    rtf_content = file.read()
                    text = rtf_converter.rtf_to_txt(rtf_content)

            processed_texts.append(text)
        except (zipfile.BadZipFile, OSError, UnicodeDecodeError) as e:  # Catch more specific errors
            print(f"Error processing {file_path}: {e}. Skipping...")

    # 2. Initialize Hugging Face Transformer Model and Tokenizer
    model_name = "w601sxs/b1ade-embed"  # "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForMaskedLM.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # 3. Generate Embeddings
    embeddings = []
    for doc_text in processed_texts:
        if not doc_text.strip():
            continue

        inputs = tokenizer(doc_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        embeddings.append(last_hidden_state.mean(dim=1).numpy().squeeze())

    if not embeddings:
        print("Error: No valid embeddings were generated. Please check your files.")
    else:
        # 4. Calculate Similarity Matrix
        similarity_matrix = cosine_similarity(embeddings)

        # 5. Create DataFrame
        filenames = [os.path.basename(file_path) for file_path in doc_files]
        df_similarity = pd.DataFrame(similarity_matrix, index=filenames, columns=filenames)

        # Display or save the similarity matrix
        print(df_similarity)

        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        df_similarity.to_csv(f'similarity_matrix_{timestamp}.csv')
