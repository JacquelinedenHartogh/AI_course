import nltk
from nltk.tokenize import word_tokenize
import string
import sys, os, glob
import numpy as np

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens
                    
    #I commented out the below sentences as I don't need it
    # Compute IDF values across sentences
    #idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, file_idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print('---')
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    #initialise dictionary
    all_files = dict()
     
    #loop over each file 
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:
             all_files[filename] = f.read().replace('\n',' ')

    return all_files
    
def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    #get all tokens
    tokens = word_tokenize(document)
    
    #make sure all upper cases are lower cases
    tokens = [w.lower() for w in tokens]    

    #load English stopwords and puncuation characters
    punc_char = string.punctuation
    stopw = nltk.corpus.stopwords.words("english")

    #remove English stopwords and puncuation characters
    words = [w for w in tokens if w not in stopw]    
    words = [w for w in words if w not in punc_char]   

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    #initialise dictionary
    IDF_values = dict()
    countW = dict()
    
    #find number of docs in documents
    num_docs = len(documents.keys())       
    
    #loop over docs and word in docs to find how often the word occurs
    #use set to remove duplicates 
    
    for doc in documents.keys():
        ListWords = set(documents[doc])
        for word in ListWords:
           
            #Plus 1 if word is in dictionary
            if word in countW.keys():
               countW[word] += 1
              
            #Set to 1 if word wasn't in dictionary yet
            else:
               countW[word] = 1
     
    #Now calculate IDF value for each word in IDF_values dictionary        
    for word in countW.keys():
        IDF_values[word] = np.log(num_docs / countW[word])
        
    return IDF_values
    

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    #initialise dictionary for ranking
    ranking = dict()
    
    #loop over all files in files and initialise key in dictionary for each file
    for file_ in files:
        ranking[file_] = []
        
        #loop over each word in query and if the word in is the file, 
        #and append IDF value
        sum_file = 0
        for word in query:    
                idf = idfs[word]
                tf = files[file_].count(word)
                sum_file += (tf*idf)
        ranking[file_] = sum_file
           
    #sort the sums so that the first item in list
    #has the highest sum               
    sorted_ranking = sorted(ranking.keys(), key=lambda x: ranking[x], reverse=True) 

    return sorted_ranking[0:n]
      

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    
    #initialise dictionary for ranking
    ranking = dict()
    
    #loop over all sentences and initiate arrays:   
    for sentence in sentences:     
        IDFvalue, counts = 0, 0       
        words = sentences[sentence]
        NrWs = len(words)
              
        #loop over words in query and set occurance + idf value for each word
        for Qword in query:
            if Qword in words:
                counts += words.count(Qword)
                IDFvalue += idfs[Qword]
        
        #calculate term density        
        TermD = counts / NrWs     
        
        #add results to dictionary for each sentence   
        ranking[sentence] = [IDFvalue, TermD]

    #sort results, by IDF value and term density
    sorted_ranking = sorted(ranking.keys(), key=lambda x: ranking[x], reverse=True) 

    return sorted_ranking[0:n]


if __name__ == "__main__":
    main()
