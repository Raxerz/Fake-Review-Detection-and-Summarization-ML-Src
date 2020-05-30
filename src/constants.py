import os

DATASET_PATH = '../datasets'

GOLD_REVIEWERS = os.path.join(DATASET_PATH, 'goldreviewers.txt')

AMAZON_REVIEWERS_PATH = os.path.join(DATASET_PATH, 'amazon.csv')

ML_PATH = os.path.join(DATASET_PATH, 'ML')

ML_GENERATED_CSV_PATH = os.path.join(DATASET_PATH, 'ML-Generated-CSVs')

ML_MODELS_PATH = os.path.join(ML_PATH, 'ML-Models')

BRANDS_PATH = os.path.join(DATASET_PATH, 'Brands')

REVIEWS_PATH = os.path.join(DATASET_PATH, 'Reviews')

REVIEWS_PARSED_PATH = os.path.join(REVIEWS_PATH, 'Parsed')

BRANDS_RAW_PATH = os.path.join(BRANDS_PATH, 'Raw')

BRANDS_PARSED_PATH = os.path.join(BRANDS_PATH, 'Parsed')

REVIEWS_RAW_PATH = os.path.join(REVIEWS_PATH, 'Raw')
