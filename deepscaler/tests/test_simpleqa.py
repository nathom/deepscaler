"""
Pytest tests for SimpleQA dataset implementation in DeepScaler.

These tests cover:
1. Dataset type definitions
2. Dataset download and processing
3. Reward function evaluation (optional, runs only if dependencies are available)
"""

import os
import sys
import json
import pytest
from pathlib import Path

from deepscaler.data import TrainDataset, TestDataset
from deepscaler.data.preprocess.simpleqa import process_simpleqa
from deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END

# Global variables to avoid redownloading for each test
TRAIN_PATH = None
TEST_PATH = None


@pytest.fixture(scope="module")
def simpleqa_dataset():
    """Fixture that ensures the SimpleQA dataset is downloaded and processed."""
    global TRAIN_PATH, TEST_PATH
    
    # Get paths relative to this test file
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(current_dir, "data/train/simpleqa.json")
    test_path = os.path.join(current_dir, "data/test/simpleqa.json")
    
    TRAIN_PATH = train_path
    TEST_PATH = test_path
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        wiki_count, total_count = process_simpleqa()
        print(f"Processed SimpleQA dataset with {wiki_count} Wikipedia questions out of {total_count} total")
    
    # Load the datasets to return them
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)
        
    return {
        "train": train_data,
        "test": test_data,
        "train_path": train_path,
        "test_path": test_path
    }


def test_dataset_enums():
    """Test that SimpleQA is properly defined in dataset enums."""
    # Check if SIMPLEQA is in both TrainDataset and TestDataset enums
    assert 'SIMPLEQA' in [ds.name for ds in TrainDataset], "SIMPLEQA not found in TrainDataset enum"
    assert 'SIMPLEQA' in [ds.name for ds in TestDataset], "SIMPLEQA not found in TestDataset enum"
    
    # Verify the enum values
    assert TrainDataset.SIMPLEQA.value == 'SIMPLEQA', f"Expected 'SIMPLEQA', got '{TrainDataset.SIMPLEQA.value}'"
    assert TestDataset.SIMPLEQA.value == 'SIMPLEQA', f"Expected 'SIMPLEQA', got '{TestDataset.SIMPLEQA.value}'"


def test_dataset_processing(simpleqa_dataset):
    """Test the SimpleQA dataset processing."""
    # Verify the dataset was processed correctly
    assert len(simpleqa_dataset["train"]) > 0, "Training dataset is empty"
    assert len(simpleqa_dataset["test"]) > 0, "Test dataset is empty" 
    
    # Verify the first entry has the expected fields
    first_train_entry = simpleqa_dataset["train"][0]
    assert "problem" in first_train_entry, "Training entry missing 'problem' field"
    assert "answer" in first_train_entry, "Training entry missing 'answer' field"
    assert "topic" in first_train_entry, "Training entry missing 'topic' field"
    assert "answer_type" in first_train_entry, "Training entry missing 'answer_type' field"
    
    # Verify the train/test split ratio (~80/20)
    total_examples = len(simpleqa_dataset["train"]) + len(simpleqa_dataset["test"])
    train_ratio = len(simpleqa_dataset["train"]) / total_examples
    assert 0.75 <= train_ratio <= 0.85, f"Training set ratio {train_ratio} is outside expected range (0.75-0.85)"


@pytest.mark.optional
def test_reward_function():
    """
    Test the SimpleQA reward function.
    
    This test is marked as optional since it requires additional dependencies
    (OpenAI API key, pylatexenc, etc.) that might not be available in all environments.
    """
    try:
        from deepscaler.rewards.reward_types import RewardConfig, RewardInput, RewardType
        from deepscaler.rewards.qa_reward import RewardQAFn
    except ImportError:
        pytest.skip("Required reward function modules not available")
    
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    # Sample question and answers
    question = "Who was the first woman to win a Nobel Prize?"
    correct_answer = "Marie Curie"
    ground_truth = {"answer": correct_answer}
    
    # Create model responses
    correct_response = f"{THOUGHT_DELIMITER_START}I need to recall who was the first female Nobel laureate. Marie Curie won the Nobel Prize in Physics in 1903 and later in Chemistry in 1911, making her the first woman to win a Nobel Prize.{THOUGHT_DELIMITER_END}\nThe first woman to win a Nobel Prize was Marie Curie."
    incorrect_response = f"{THOUGHT_DELIMITER_START}I believe the first woman to win a Nobel Prize was Rosalind Franklin for her work on DNA.{THOUGHT_DELIMITER_END}\nThe first woman to win a Nobel Prize was Rosalind Franklin."
    
    # Initialize reward function
    config = RewardConfig(correct_reward=1.0, incorrect_reward=0.0)
    reward_fn = RewardQAFn(config)
    
    # Test with correct answer
    input_correct = RewardInput(
        problem=question,
        model_response=correct_response,
        problem_type=RewardType.QA,
        ground_truth=ground_truth
    )
    
    result = reward_fn(input_correct)
    assert result.reward == 1.0, f"Expected reward 1.0 for correct answer, got {result.reward}"
    assert result.is_correct is True, "Expected is_correct=True for correct answer"
    
    # Test with incorrect answer
    input_incorrect = RewardInput(
        problem=question,
        model_response=incorrect_response,
        problem_type=RewardType.QA,
        ground_truth=ground_truth
    )
    
    result = reward_fn(input_incorrect)
    assert result.reward == 0.0, f"Expected reward 0.0 for incorrect answer, got {result.reward}"
    assert result.is_correct is False, "Expected is_correct=False for incorrect answer"


if __name__ == "__main__":
    pytest.main(["-v", __file__])