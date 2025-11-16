"""
Data loading and preprocessing pipeline.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Tuple
import logging

from src.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.logs_dir / 'data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_and_merge_datasets(file_paths=None) -> pd.DataFrame:
    """
    Load and merge multiple CSV datasets.
    
    Args:
        file_paths: List of paths to CSV files
    
    Returns:
        Merged DataFrame
    """
    if file_paths is None:
        file_paths = config.data.raw_data_paths
    
    logger.info(f"Loading {len(file_paths)} datasets...")
    
    dataframes = []
    for path in file_paths:
        try:
            df = pd.read_csv(path, encoding=config.data.encoding)
            logger.info(f"Loaded {path}: {df.shape}")
            
            # Ensure required columns
            if 'text' not in df.columns:
                df['text'] = pd.NA
            if 'title' not in df.columns:
                df['title'] = pd.NA
            
            dataframes.append(df[["title", "class", "text"]])
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            raise
    
    # Merge datasets
    merged = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Merged dataset shape: {merged.shape}")
    
    return merged


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize inconsistent labels to binary (0=Fake, 1=True).
    
    Args:
        df: Input DataFrame with 'class' column
    
    Returns:
        DataFrame with normalized labels
    """
    df = df.copy()
    
    # Convert to lowercase
    df['class'] = df['class'].astype(str).str.lower()
    
    # Mapping rules
    true_labels = ['true', '1', 'mostly-true', 'barely-true']
    false_labels = ['false', '0']
    exclude_labels = ['half-true', 'pants-fire', 'half-flip', 'no-flip', 'full-flop']
    
    # Map to binary
    df['class'] = df['class'].apply(lambda x: 
        '1' if x in true_labels else ('0' if x in false_labels else None)
    )
    
    # Remove excluded and unmapped
    initial_len = len(df)
    df = df[~df['class'].isin(exclude_labels)]
    df = df.dropna(subset=['class'])
    removed = initial_len - len(df)
    
    logger.info(f"Label normalization: {removed} samples excluded")
    logger.info(f"Final distribution: {df['class'].value_counts().to_dict()}")
    
    return df


def preprocess_text(df: pd.DataFrame, text_column: str = 'title') -> pd.DataFrame:
    """
    Clean and preprocess text data.
    
    Args:
        df: Input DataFrame
        text_column: Column containing text to clean
    
    Returns:
        DataFrame with 'clean_text' column
    """
    import re
    import string
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    
    # Combine stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(ENGLISH_STOP_WORDS)
    stop_words.update(config.preprocessing.custom_stopwords)
    
    logger.info(f"Preprocessing text with {len(stop_words)} stopwords...")
    
    def clean_text(text):
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = ' '.join(text.split())
        
        if config.preprocessing.remove_stopwords:
            tokens = [t for t in text.split() 
                     if t not in stop_words and len(t) >= config.preprocessing.min_token_length]
            text = ' '.join(tokens)
        
        return text.strip()
    
    df = df.copy()
    df['clean_text'] = df[text_column].apply(clean_text)
    
    # Remove empty texts
    initial_len = len(df)
    df = df[df['clean_text'].str.len() > 0]
    removed = initial_len - len(df)
    
    if removed > 0:
        logger.warning(f"Removed {removed} empty texts after cleaning")
    
    return df


def run_data_pipeline() -> pd.DataFrame:
    """
    Execute complete data pipeline.
    
    Returns:
        Preprocessed DataFrame ready for model training
    """
    logger.info("=" * 60)
    logger.info("Starting Data Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Load and merge
    df = load_and_merge_datasets()
    
    # Step 2: Normalize labels
    df = normalize_labels(df)
    
    # Step 3: Preprocess text
    df = preprocess_text(df)
    
    # Step 4: Save processed data
    output_path = config.data.output_path
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")
    
    logger.info("=" * 60)
    logger.info("Data Pipeline Complete")
    logger.info(f"Final dataset: {df.shape}")
    logger.info("=" * 60)
    
    return df


if __name__ == "__main__":
    run_data_pipeline()
