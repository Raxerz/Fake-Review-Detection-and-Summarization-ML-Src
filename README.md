Steps to set up the environment and run the summarization and fake review detection system

Prerequisites:
Download the amazon review and brands dataset from "http://jmcauley.ucsd.edu/data/amazon/" and store it in "Fake-review-detection-and-Summarization/datasets/Gzips/" and "Fake-review-detection-and-Summarization/datasets/Brands/" respectively

Steps to run this system:

1. cd Fake-review-detection-and-Summarization-ML-src
2. pip -r install requirements.txt
3. Install the nltk packages mentioned in the "nltkpackages" file
4. Run extractBrands.py to generate the brands pickle file.
5. Run parseReviews.py to generate the pickle file for the reviews.
6. Run Launcher.py to start execution
