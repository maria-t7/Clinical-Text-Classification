# Clinical Text Classification

## Overview
The goal of this project is to classify medical transcriptions into their corresponding 
medical specialties using natural language processing and machine learning.

## Dataset
- Source: MTSamples (mtsamples.com)
- Total samples: 4999
- Original classes: 40
- Classes used: Those with minimum 70 samples (ensures at least 14 test samples per class after 80/20 split)
- Input: transcription column
- Output: medical_specialty label

## Preprocessing
- Lowercase conversion
- Medical section header removal (SUBJECTIVE:, ASSESSMENT:, PLAN: etc.)
- Punctuation and special character removal (numbers retained for clinical significance)
- Word tokenization using NLTK word_tokenize()
- Part-of-Speech (POS) tagging for context-aware lemmatization
- Stopword removal
- Lemmatization with POS context

## Feature Extraction
- TF-IDF Vectorizer (max_features=1000, ngram_range=(1,1), min_df=5)
- 80/20 stratified train-test split

## Models Evaluated
- Logistic Regression (GridSearchCV tuned)
- Linear SVC (GridSearchCV tuned)
- Naive Bayes
- Passive Aggressive Classifier
- Random Forest Classifier
- XGBoost Classifier

## Results
| Model | Train | Test | Gap | 95% CI |
|-------|-------|------|-----|--------|
| Logistic Regression | 0.4713 | 0.4147 | 0.0566 | (0.3825, 0.4470) |
| LinearSVC | 0.4618 | 0.4047 | 0.0571 | (0.3726, 0.4368) |
| Naive Bayes | 0.4364 | 0.4002 | 0.0362 | (0.3682, 0.4323) |
| Passive Aggressive | 0.5809 | 0.1483 | 0.4326 | (0.1250, 0.1715) |
| Random Forest | 0.5917 | 0.1371 | 0.4546 | (0.1146, 0.1596) |

## Deployment
The app is deployed on Streamlit Cloud.
[Click here to use the app](YOUR_STREAMLIT_URL)

## Future Work
- Combine transcription and description as input
- Explore BioBERT or ClinicalBERT for better medical text understanding
- Increase training data for underrepresented specialties

## Author
Maria Rahman Tasnim
