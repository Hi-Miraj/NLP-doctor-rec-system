# NLP-doctor-rec-system
# NLP-Based Doctor Recommendation System

This repository contains a machine learning model that recommends a doctor type based on user-input symptoms, using Natural Language Processing (NLP) techniques. The core approach involves cleaning and vectorizing symptom text data, encoding doctor types, and training a Logistic Regression classifier for recommendation.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Methods](#methods)
- [File Output](#file-output)
- [Dependencies](#dependencies)
- [License](#license)

## Project Overview

This project leverages NLP techniques to map symptoms to appropriate doctor types using a supervised learning approach. It uses text preprocessing, TF-IDF vectorization, and a Logistic Regression model to classify symptoms into predefined doctor categories.

## Dataset

The model is trained on a dataset containing medical symptoms (`symptoms`) and corresponding doctor types (`doctortype`). The dataset is loaded from an Excel file (`med depart.xlsx`) and requires columns named:
- **symptoms**: Free text description of symptoms.
- **doctortype**: Type of doctor suitable for these symptoms.

### Data Processing

Each symptom entry undergoes:
1. **Text cleaning**: Removing non-alphanumeric characters, converting to lowercase, and removing stopwords.
2. **Label Encoding**: Encoding doctor types to integer labels.
3. **TF-IDF Transformation**: Transforming cleaned symptoms to a TF-IDF matrix for feature extraction.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/nlp-based-doctor-recommendation-system.git
    ```
2. Navigate to the project directory:
    ```bash
    cd nlp-based-doctor-recommendation-system
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Ensure that `med depart.xlsx` is placed in the directory specified in the code (`E:\Miraj Shafek things\med depart.xlsx`) or adjust the path accordingly.

## Usage

To run the model, execute the main script. Input symptoms in the console, and the model will output a recommended doctor type.

```bash
python doctor_recommendation.py
```

**Interactive Session**:
- The script will prompt for symptoms input.
- Enter symptoms as a text string. Type `exit` to end the program.

## Code Structure

### 1. Text Preprocessing

Text data is preprocessed to prepare it for vectorization:
- **`clean_text` function**: Handles the cleaning of raw symptom text, removing non-alphanumeric characters, converting text to lowercase, and removing common English stopwords.

### 2. Model Training

The code uses Logistic Regression as the classification model:
- **Vectorization**: Symptom text is transformed using `TfidfVectorizer`.
- **Label Encoding**: Doctor types are encoded into integer labels using `LabelEncoder`.
- **Training**: The Logistic Regression model is trained on 80% of the dataset, with 20% reserved for testing.

### 3. Doctor Recommendation Function

The `recommend_doctor` function:
- Cleans and vectorizes user-input symptoms.
- Uses the trained model to predict a doctor type.
- Decodes the prediction back to a readable doctor type string.

## Methods

### Function: `clean_text`
- **Parameters**: `text` (str) - Raw symptom text.
- **Returns**: Cleaned, tokenized text without stopwords.

### Function: `recommend_doctor`
- **Parameters**: `symptoms` (str) - User-input symptoms.
- **Returns**: Recommended doctor type (str) - The model’s prediction.

### Main Script
- The main script captures user input, runs `recommend_doctor`, and outputs the recommendation both in the console and in a JSON file (`recommendation.json`).

## File Output

The recommendation result is saved in `recommendation.json` with the structure:
```json
{
    "symptoms": "User-input symptoms",
    "recommended doctor": "Predicted doctor type"
}
```

## Dependencies

- `nltk`: For stopword removal.
- `pandas`: For data handling.
- `scikit-learn`: For Label Encoding, TF-IDF vectorization, and Logistic Regression.
- `re`: For text cleaning via regex.
- `json`: For writing output to JSON.

Install all dependencies using `pip install -r requirements.txt`.

## License

This project is licensed under the MIT License.

---
