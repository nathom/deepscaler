"""
Preprocessing script for SimpleQA dataset.

This script:
1. Downloads the SimpleQA CSV dataset from the internet
2. Filters for questions whose answers can be found on Wikipedia
3. Converts to the JSON format used by DeepScaler
4. Creates parquet files for direct use in training
"""

import argparse
import csv
import json
import os
import random
import io
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

# Get the current file's directory
current_dir = Path(__file__).parent.absolute()
data_dir = current_dir.parent

# URL to the SimpleQA test set
SIMPLEQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"

# Output file paths for JSON files (intermediate format)
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

def process_raw_examples(examples: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    """
    Process raw SimpleQA examples into the format expected by DeepScaler training.
    
    Args:
        examples: List of SimpleQA examples
        split: Dataset split ('train' or 'test')
        
    Returns:
        List of processed examples
    """
    processed_data = []
    
    for idx, example in enumerate(examples):
        question = example.get('problem', '')
        answer = example.get('answer', '')
        topic = example.get('topic', 'Unknown')
        answer_type = example.get('answer_type', 'Unknown')
        
        # Add instruction to the question
        instruction = "Answer this factual question accurately and concisely."
        question = f"{question} {instruction}"
        
        # Create structured data for training
        data = {
            "data_source": "simpleqa",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "qa",
            "reward_model": {
                "style": "qa",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'topic': topic,
                'answer_type': answer_type
            }
        }
        
        processed_data.append(data)
    
    return processed_data

def process_simpleqa(output_dir: Optional[str] = None):
    """
    Process the SimpleQA dataset:
    1. Download the dataset from the internet
    2. Filter for Wikipedia-sourced questions
    3. Convert to DeepScaler JSON format
    4. Split into training and test sets
    5. Create parquet files for training
    
    Args:
        output_dir: Optional directory to save parquet files
    
    Returns:
        tuple: (wiki_count, total_count)
    """
    # Set output directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        # Default to data_dir (deepscaler/data) - parent of this script's directory
        out_dir = data_dir
    
    os.makedirs(out_dir, exist_ok=True)
    
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
    
    # Save to JSON files (intermediate format for dataset loading)
    with open(TRAIN_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
    
    with open(TEST_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Saved {len(train_data)} training examples to {TRAIN_OUTPUT_PATH}")
    print(f"Saved {len(test_data)} test examples to {TEST_OUTPUT_PATH}")
    
    # Process data for parquet format
    processed_train = process_raw_examples(train_data, 'train')
    processed_test = process_raw_examples(test_data, 'test')
    
    # Save to parquet files
    train_df = pd.DataFrame(processed_train)
    test_df = pd.DataFrame(processed_test)
    
    train_parquet_path = out_dir / 'simpleqa_train.parquet'
    test_parquet_path = out_dir / 'simpleqa_test.parquet'
    
    train_df.to_parquet(train_parquet_path)
    test_df.to_parquet(test_parquet_path)
    
    print(f"Saved {len(processed_train)} training examples to {train_parquet_path}")
    print(f"Saved {len(processed_test)} test examples to {test_parquet_path}")
    
    return wiki_count, total_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SimpleQA dataset for DeepScaler")
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the parquet files (defaults to deepscaler/data directory)')
    args = parser.parse_args()
    
    process_simpleqa(args.output_dir)