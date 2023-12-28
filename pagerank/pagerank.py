import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
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
    
    result_dict = {}
    
    if len(corpus[page]) < 1:
        for page in corpus:
            result_dict[page] = 1 / len(corpus)
            
    else:

        pages_choose_randomly = (1 - damping_factor) / len(corpus)
        
        num_page_connected = len(corpus[page])
        
        for i in corpus:
            result_dict[i] = pages_choose_randomly
            
        for page in corpus[page]:
            result_dict[page] = damping_factor / num_page_connected + pages_choose_randomly

    return result_dict
    

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to the transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    prob_dict = {page: 0 for page in corpus}
    
    lst_pages = list(corpus.keys())
    
    # get random first page
    rand_current_page = random.choice(lst_pages)
    
    prob_dict[rand_current_page] += 1
    

    
    # weights = list(prob_next.values())
    
    # randomly choose next page after getting the list prob from the transition model
    # rand_pages = random.choices(lst_pages, weights=weights, k=n)
    
    for i in range(1, n):
        # get prob to move next page after the first page
        prob_next = transition_model(corpus, rand_current_page, damping_factor)
        
        # Convert probabilities to weights
        weights = list(prob_next.values())
        
        # randomly choose the next page based on the weights
        rand_next_page = random.choices(lst_pages, weights=weights, k=1)[0]
        
        # Update the count for the chosen page
        prob_dict[rand_next_page] += 1
        
        # Update the current page for the next iteration
        rand_current_page = rand_next_page
        
        
        
        
    
    
    # Normalize values to make them sum to 1
    prob_dict = {key: value / n for key, value in prob_dict.items()}
    
    total_probability = sum(prob_dict.values()) 
    
    # check if total probability is 1
    
    # print(prob_dict)
    # print("Total Probability:", total_probability)
    
    return prob_dict



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    ranks = {}
    threshold = 0.0005
    N = len(corpus)
    
    for page in corpus:
        ranks[page] = 1 / N

    while True:
        count = 0
        new_ranks = {}
        
        for page in corpus:
            new_rank = (1 - damping_factor) / N
            sigma = 0

            for page_nxt in corpus:
                if page in corpus[page_nxt]:
                    num_links = len(corpus[page_nxt])
                    sigma += ranks[page_nxt] / num_links

            new_rank += damping_factor * sigma

            if abs(ranks[page] - new_rank) < threshold:
                count += 1

            new_ranks[page] = new_rank

        if count == N:
            break

        ranks = new_ranks

    return ranks


if __name__ == "__main__":
    main()
