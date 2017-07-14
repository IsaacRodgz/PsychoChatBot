# Import various modules for string cleaning
# -*- coding: utf-8 -*- 
import sys
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
from unidecode import unidecode
from nltk.corpus import words

def text_to_wordlist( text, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.

    # 1. Remove HTML and convert words to lower case
	text_ = BeautifulSoup(text, "html.parser").get_text().lower()
  
    # 2. Filter words including words with accents and special symbols
	try:
		#text_ = re.search(re.compile('((\w+\s)|(\w+\W+\w+\s))+', re.UNICODE), u"{0}".format(text_)).group(0)
		words = re.findall(r'(\w+)', u"{0}".format(text_), re.UNICODE)
	except AttributeError:
		pass

    # 3. Convert words to lower case and split them
	#words = text_.lower().split()

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
	#text = unidecode(text)
	#text.encode("ascii")
	#print('~~~~~~~~~~ascii~~~~~~~~~~~')
	#print(text)

	# 1. Use the NLTK tokenizer to split the paragraph into sentences
	raw_sentences = tokenizer.tokenize(text.strip())

    # 2. Loop over each sentence
	sentences = []
	for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
		if len(raw_sentence) > 0:
            # Otherwise, call text_to_wordlist to get a list of words
			sentences.append( text_to_wordlist( raw_sentence, remove_stopwords ))

    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
	return sentences

def extract_wiki(size):
	# Parameter size limits the number of sentences to extract from file

	#NLTK tokenizer to split the paragraph into sentences
	tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

	# Initialize an empty list of sentences
	sentences = []

	# Specific words to delete from text
	not_words = {'ref', 'http', 'url', 'www', 'html', 'p', 'align', 'left', 'dq', 'ii', 'pp', 'hl', 'br', 'sp',
				 'aspx','https', 'com', 'small', 'span', 'class', 'nbsp', 'org', 'fechaacceso', 'fechaarchivo',
				 'pg', 'htm', 'gc', 'pmid', 'doi', 'eurostat', 'esd', 'php', 'isbn', 'gov', 'co', 'harvnp', 'books',
				 'urlarchivo', 'az', 'º', 'xix', 'studies', 'ee', 'ua', 'saudi', 'von', 'greece', 'classora',
				 'gr', 'factbook', 'shtml', 'lang', 'hr', 'efe', 'gould', 'allmusic', 'iii', 'nº', 'onepage',
				 'stm', 'du', 'elpais', 'ue', 'files', 'cfm', 'facts', 'au', 'ec', 'ppa', 'xvii', 'nih', 'der',
				 'ved', 'iv', 'lpg', 'infobae', 'oecd', 'reports', 'des', 'com_content', 'bhutan', 'svg', 'itemid',
				 'nations', 'elmundo', 'wp', 'll', 'nytimes', 'ru', 'states', 'cr', 'fr', 'tls', 'pdfs', 'jsp',
				 'reuters', 'ª', 'boe', 'tes', 'ch', 'títulotrad', 'undp', 'bgcolor', 'drae', 'pt', 'eng', 'xvi',
				 'edu', 'km²', 'cl', 'et', 'iaaf', 'psoe', 'pr', 'harvsp', 'issn', 'cgi', 'ft', 'pages', 'int', 'ac',
				 'pe', 'xiii', 'nsf', 'mw', 'vii', 'harvnb', 'ph', 'ei', 'xviii', 'archiveurl', 'elcomercio', 'rae',
				 'ocde', 'anos', 'tolkien', 'añoacceso', 'abc', 'il', 'pubs', 'ix', 'sci', 'scielo', 'results',
				 'fao', 'gt', 'emol', 'nz', 'oclc', 'viii', ''}

	# Delete english words
	english_vocab = set(w.lower() for w in words.words())

	# Counter of sentences extracted from file
	count = 0

	# Open spanish wikipedia file
	print("\n\n")
	with open('C:/Users/irb/Downloads/eswiki-latest-pages-articles/eswiki-latest-pages-articles.xml', 'r', encoding='utf-8') as file:
		# For each sentence in the file
		for sentence in file:
			if len(sentence) > 1:

				# Transform sentence to list of words
				wordlist = text_to_sentences(sentence, tokenizer)

				# Indexes of words to delete from wordlist
				not_words_index = []

				# Search for words to delete and save indexes in list not_words_index
				# Delete words in not_words, numbers and english words
				for i in range(len(wordlist)):
					for j in range(len(wordlist[i])):
						word = wordlist[i][j]
						if (word in not_words) or (re.search("[0-9]", word)) or (word in english_vocab):
							not_words_index.append((i,j))

				# Delete words with the help of not_words_index 
				if len(not_words_index) > 0:
					for i,j in sorted(not_words_index, reverse=True):
						del wordlist[i][j]

				# Empty list, ready for the next wordlist
				not_words_index = []

				if len(wordlist) > 0:
					# Filter wordlists with at least 50 words
					if len(wordlist[0]) > 50:

						try:
							# print('~~~~~~~~~~raw~~~~~~~~~~')
							# print(sentence)

							# print('~~~~~~~~~~wordlist~~~~~~~~~~')
							# print(wordlist)
							# print("\n\n")
							#input()

							print(count)

							# Add wordlist to corpus
							sentences += wordlist

							# Increment counter of sentences extracted from file
							count += 1

							# If size sentences are extracted form file, finish for loop
							if count > size:
								break
						
						except UnicodeEncodeError:
							pass
							#print('~~~~~~~~~~raw~~~~~~~~~~')
							#print(u"{}".format(sentence).encode(sys.stdout.encoding, errors='replace'))
	return sentences