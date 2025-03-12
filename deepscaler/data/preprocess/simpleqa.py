"""
Preprocessing script for SimpleQA dataset.

This script downloads the SimpleQA CSV dataset from the internet, converts it to 
the JSON format used by DeepScaler, and filters for questions whose answers
can be found on Wikipedia.
"""

import csv
import json
import os
import random
import io
import requests
from pathlib import Path

# Get the current file's directory
current_dir = Path(__file__).parent.absolute()
data_dir = current_dir.parent

# URL to the SimpleQA test set
SIMPLEQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"

# Output file paths
TRAIN_OUTPUT_PATH = data_dir / "train" / "simpleqa.json"
TEST_OUTPUT_PATH = data_dir / "test" / "simpleqa.json"

# Create output directories if they don't exist
os.makedirs(data_dir / "train", exist_ok=True)
os.makedirs(data_dir / "test", exist_ok=True)

def download_simpleqa_dataset():
    """
    Download the SimpleQA dataset from the internet.
    
    Returns:
        str: CSV content as a string
    """
    print(f"Downloading SimpleQA dataset from {SIMPLEQA_URL}")
    try:
        response = requests.get(SIMPLEQA_URL)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.text
    except Exception as e:
        raise RuntimeError(f"Failed to download SimpleQA dataset: {e}")

def process_simpleqa():
    """
    Process the SimpleQA dataset:
    1. Download the dataset from the internet
    2. Filter for Wikipedia-sourced questions
    3. Convert to DeepScaler JSON format
    4. Split into training and test sets
    """
    # Download the dataset
    csv_content = download_simpleqa_dataset()
    
    wiki_questions = []
    total_count = 0
    wiki_count = 0
    
    # Parse the CSV content
    reader = csv.reader(io.StringIO(csv_content))
    header = next(reader)  # Skip header row
    
    for row in reader:
        total_count += 1
        metadata_str = row[0]
        problem = row[1]
        answer = row[2]
        
        # Check if 'wikipedia' is mentioned in the metadata
        if 'wikipedia' in metadata_str.lower():
            wiki_count += 1
            
            # Extract topic and answer_type from metadata
            try:
                # Parse the metadata string into a dictionary
                # It's in the format: "{'topic': 'X', 'answer_type': 'Y', 'urls': [...]}"
                metadata_dict = eval(metadata_str)
                topic = metadata_dict.get('topic', 'Unknown')
                answer_type = metadata_dict.get('answer_type', 'Unknown')
            except (SyntaxError, NameError):
                topic = 'Unknown'
                answer_type = 'Unknown'
            
            # Create entry in DeepScaler format
            entry = {
                "problem": problem,
                "answer": answer,
                "topic": topic,
                "answer_type": answer_type
            }
            
            wiki_questions.append(entry)
    
    print(f"Total questions: {total_count}")
    print(f"Questions with Wikipedia as source: {wiki_count}")
    print(f"Percentage: {(wiki_count/total_count)*100:.2f}%")
    
    # Shuffle the data with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(wiki_questions)
    
    # Split into 80% train, 20% test
    split_idx = int(len(wiki_questions) * 0.8)
    train_data = wiki_questions[:split_idx]
    test_data = wiki_questions[split_idx:]
    
    # Save to JSON files
    with open(TRAIN_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
    
    with open(TEST_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Saved {len(train_data)} training examples to {TRAIN_OUTPUT_PATH}")
    print(f"Saved {len(test_data)} test examples to {TEST_OUTPUT_PATH}")
    
    return wiki_count, total_count

if __name__ == "__main__":
    process_simpleqa()