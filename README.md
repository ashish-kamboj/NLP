### NLP/Text Analytics/Text Mining

**1. Text Cleaning** - It involves cleaning the text in following ways:
  - **Remove words** - If the data is extracted using web scraping, you might want to remove html tags.
  - **Remove stop words** - Stop words are a set of words which helps in sentence construction and don't have any real information. Words such as a, an, the, they, where etc. are categorized as stop words.
  - **Convert to lower** - To maintain a standarization across all text and get rid of case differences and convert the entire text to lower.
  - **Remove punctuation** - We remove punctuation since they don't deliver any information.
  - **Remove number** - Similarly, we remove numerical figures from text
  - **Remove whitespaces** - Then, we remove the used spaces in the text.
  - **Stemming & Lemmatization** - Finally, we convert the terms into their root form. For example: Words like playing, played, plays gets converted to the root word 'play'. It helps in capturing the intent of terms precisely.

<br><br>
**2. Feature Engineering** 
  - **n-grams:** The idea behind this technique is to explore the chances that when one or two or more words occurs together gives more information to the model.
  - **TF-IDF:** It is also known as Term Frequency - Inverse Document Frequency. This technique believes that, from a document corpus, a learning algorithm gets more information from the rarely occurring terms than frequently occurring terms.  Using a weighted scheme, this technique helps to score the importance of terms.
  - **Cosine Similarity:** This measure helps to find similar documents.

<br><br>
**3. Model Building**
  - Navie Bayes
  - SVM
  - Topic Modeling
  - Name-Entity Recognition
