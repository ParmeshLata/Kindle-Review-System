# Kindle Review Sentiment Analysis

This project is a **sentiment analysis system** for Kindle reviews using **Naïve Bayes classification**. It processes reviews, removes noise, and classifies them as **positive or negative**. The project is built using **Python, Streamlit, and scikit-learn**.

## Features
- **Preprocesses** text by removing special characters, stopwords, and URLs.
- **Trains a Naïve Bayes classifier** on review data.
- **Classifies reviews** as positive or negative.
- **Interactive UI** using Streamlit.

## Installation
1. **Clone the repository**:
   ```sh
   git clone https://github.com/ParmeshLata/Kindle-Review-System.git
   cd Kindle-Review-System
   ```

2. **Install dependencies**:
  - NLTK Library
  - Streamlit
  - Scikit-Learn
  - Pandas
  - RE

3. **Download NLTK stopwords** (if not already downloaded):
   ```python
   import nltk
   nltk.download("stopwords")
   ```

## Usage
### **Run the Streamlit App**
```sh
streamlit run kindle_review.py
```
This will launch the web interface for entering and analyzing Kindle reviews.

## License
I have developed this project under the learning of https://github.com/krishnaik06 and credit for dataset goes to him only.
