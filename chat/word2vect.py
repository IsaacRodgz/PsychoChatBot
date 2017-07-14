# Import various modules for string cleaning
# -*- coding: utf-8 -*- 
from bs4 import BeautifulSoup
import re
import sys
from nltk.corpus import stopwords
import nltk.data
from unidecode import unidecode
from nltk.corpus import words
import wiki_parse

def text_to_wordlist( text, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.

    # 1. Remove HTML
	text_ = BeautifulSoup(text, "html.parser").get_text()
  
    # 2. Remove non-letters
	text_ = re.sub("[^a-zA-Z]"," ", text_)

    # 3. Convert words to lower case and split them
	words = text_.lower().split()

   	# 4. Optionally remove stop words (false by default)
	if remove_stopwords:
		stops = set(stopwords.words("spanish"))
		words = [w for w in words if not w in stops]

    # 5. Return a list of words
	return(words)

# Define a function to split a review into parsed sentences
def text_to_sentences( text, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words

    # 0. Replace accents with equivalent ASCII
	text = unidecode(text)
	text.encode("ascii")

	# 1. Use the NLTK tokenizer to split the paragraph into sentences
	raw_sentences = tokenizer.tokenize(text.strip())

    # 2. Loop over each sentence
	sentences = []
	for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
		if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
			sentences.append( text_to_wordlist( raw_sentence, remove_stopwords ))

    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
	return sentences


tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

sentences = []  # Initialize an empty list of sentences

# print("\n\n")
# with open('mexico_dataset.txt', 'r', encoding='utf-8') as file:
# 	for sentence in file:
# 		if len(sentence) > 1:
# 			wordlist = text_to_sentences(sentence, tokenizer)
# 			sentences += wordlist
# 			#print("\n**********\n")
# 			#print(wordlist)
# 			#input()

print("\nParsing wiki")
# Extract 1000 sentences
sentences = wiki_parse.extract_wiki(5000)
print("\nWiki parsed")
input()

# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)

from gensim import models
model = models.Word2Vec.load(model_name)

for word in model.wv.vocab:
	print(word)

model.wv.most_similar(positive=['woman', 'king'], negative=['man'])
model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])