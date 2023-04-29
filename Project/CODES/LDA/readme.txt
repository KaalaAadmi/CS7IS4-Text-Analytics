we need gensim library and pretrained GloVe embeddings.
python -m gensim.scripts.glove2word2vec -i glove.6B.300d.txt -o glove.6B.300d.word2vec.txt

place this file "glove.6B.300d.word2vec.txt" in the same folder as the ipynb

The code does the following:
Import necessary libraries and load the GloVe word embeddings model.
Define the mission statements and university rankings as lists.
Preprocess the mission statements by removing punctuation, tokenizing, lemmatizing, and removing stop words.
Use cross-validated search with KFold to find the optimal number of LDA topics based on the lowest perplexity score.
Perform LDA topic modeling using the optimal number of topics.
Compute Spearman's rank correlation between university rankings and LDA topic distributions for each topic.
Create average word embeddings for each mission statement using the GloVe model.
Compute Spearman's rank correlation between university rankings and word embeddings.
Display the top words for each LDA topic.
Create scatter plots for university rankings vs LDA topic distributions and university rankings vs word embeddings (first principal component) using PCA.
Create a heatmap to visualize the correlation and p-values for the LDA topics and word embeddings.
Optionally, create scatter plots for university rankings vs LDA topic distributions and university rankings vs word embeddings with different colors for each university, if university names are provided.