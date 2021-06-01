import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    
    #T=transition_model(corpus, "1.html", DAMPING)
    #print(T)
    #stop
        
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
        
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    #initiate probs dictionary
    probs = dict()  
    pages=corpus[page] 
    
    #check for each page if it's one of the links and add to standard p
    for key in list(corpus.keys()):
        #add the 0.15/N to all
        probs[key]=(1-damping_factor)/len(corpus)
        #no links? 
        if len(pages) == 0:
           probs[key]+=damping_factor/len(corpus)
        #links? than divide damping factor by number of links and add
        if key in pages:       
           probs[key]+=damping_factor/len(pages)

    return probs  


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #initiate sample dictionary    
    sampled_PR = dict()
    
    #create keys for sample dictionary
    for page in corpus:
        sampled_PR[page]=0
        
    sample=None

    #continuing until we've done all samples
    for step in range(n):
          if sample is None:
             sample = random.choices(list(corpus.keys()),k=1)[0]
          #finding the probability distribution of the last added sample
          else:
             choice=transition_model(corpus,sample,damping_factor)
             os = list(choice.keys())
             ps = list()
             for i in os:
                 ps.append(choice[i])

             sample = random.choices(population=os, weights=ps,k=1)[0]

          sampled_PR[sample] +=1

    #adding all the occurances in the sampled list to find the PR   
    for page in corpus:
            sampled_PR[page] /= n
    
    return sampled_PR

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #initiate iter and ref dictionary (ref for while-loop)     
    iter_PR = dict()
    ref_PR = dict()
    second_term = dict()       
    
    d=damping_factor
    
    #create keys for iter and ref dictionary, and set initial values
    for page in corpus:
        iter_PR[page]=1/len(corpus.keys())
        ref_PR[page]=0 
         
    count=0 
            
    while True:       
       ref_PR=copy.deepcopy(iter_PR)
       count=0

       for page1 in corpus.keys():
           second_term[page1]=(1-d)/len(corpus.keys())          
           for page2 in corpus:

               if page2 != page1 and page1 in corpus[page2]:
                  if len(corpus[page2]) == 0:
                     Numlink=len(corpus.keys())
                  else:
                     Numlink=len(corpus[page2])
                  second_term[page1] += d * iter_PR[page2]/Numlink
                  #print(second_term, page1)
                  
       for page in corpus:
           iter_PR[page]=second_term[page] 
       for page in corpus:            
           if abs(iter_PR[page] - ref_PR[page]) < 0.0005:
              count+=1
              
       if count==len(corpus.keys()):
          break
          
    return iter_PR

if __name__ == "__main__":
    main()
