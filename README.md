# news-classifier
ds4400 machine learning spring 2026

Data
Source: WELFake dataset (~72,000 labeled news articles; 0 = real, 1 = fake)
Features used: article title, article text, plus engineered features (see below)

Approach
- Text preprocessing: cleaned and tokenized article text and titles using NLTK (stopword removal, lemmatization)
- Feature engineering: created custom features including:
  - Title and text length
  - Exclamation point counts (title/text)
  - Title and text sentiment and subjectivity scores (via TextBlob)
  - A custom clickbait score feature
- Text vectorization: applied TF-IDF vectorization to article titles and text
- Modeling: built and compared four classification models:
  - Logistic Regression
  - k-Nearest Neighbors (kNN)
  - Linear Discriminant Analysis (LDA)
  - MLP Neural Network
- Evaluation: assessed models using accuracy, precision, recall, F1 score, and ROC-AUC; examined feature importance/coefficients for each model

Results
- Best-performing model: MLP Neural Network, achieving 96.7% accuracy
- The engineered clickbait score feature ranked among the top three most predictive features across models

Tools
Python, pandas, NLTK, TextBlob, scikit-learn, matplotlib
