import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    
    #of all people in people we want to calculate how likely it is that 
    # have the info as presented in the other inputs
    
    p_person=set()
    genes=dict()
    traits=dict()
    prob_give=dict()
    
    #let's fill some dictionaries, so we can easily use to values/flags
    for person in people:
           
           #figure out flag of have-trait
           if person in have_trait:
              traits[person] = True
           else:
              traits[person] =  False 
              
           #figure out number of genes, and prob to give gene
           if person in one_gene:
              genes[person]=1
              prob_give[person]=0.5              
           elif person in two_genes:
              genes[person]=2
              prob_give[person]=1-PROBS['mutation']
                            
           else: 
              genes[person]=0
              prob_give[person]=PROBS['mutation']

    #now we have all the dictionaries filled, we can use them to  calculate the probs      
    for person in people:
           p_gene, p_trait=0,0
           #figure out probabilities:
           if people[person]['mother'] == None:
              p_gene  = PROBS['gene'][genes[person]]
              p_trait = PROBS['trait'][genes[person]][traits[person]]
              p_person.add(p_gene*p_trait)
           else: 
              mother = people[person]['mother']
              gene_m=genes[mother]
              father = people[person]['father']
              gene_f=genes[father]
              if genes[person]==0:
                 p_gene=(1-prob_give[mother])*(1-prob_give[father])                 
              elif genes[person]==1:
                 p_gene=prob_give[mother]*(1-prob_give[father])
                 p_gene+=(1-prob_give[mother])*prob_give[father]
              else:
                 p_gene=prob_give[mother]*prob_give[father]                
              p_trait= PROBS['trait'][genes[person]][traits[person]]
              p_person.add(p_gene*p_trait)             
                 
    #now that we have all the probs for the people, we can calculate the final
    #prob, which is the product of all individual probs             
    prob_all_true=1
    for item in p_person:
        prob_all_true=prob_all_true*item            

    return prob_all_true


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """

    #update the probabilities dictionary with the new p:
    for person in probabilities:
           #figure out flag of have-trait, and add p
           if person in have_trait:
              probabilities[person]['trait'][True] = p
           else:
              probabilities[person]['trait'][False] = p 
              
           #figure out number of genes, and prob to give gene
           if person in one_gene:
              probabilities[person]['gene'][1]=p              
           elif person in two_genes:
              probabilities[person]['gene'][2]=p                         
           else: 
              probabilities[person]['gene'][0]=p
    

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
        
    #sum all values in probabilities-gene or -trait and normalise to 1
    for person in probabilities:
        for feature in probabilities[person]:
            sum_prob=0
            for item in probabilities[person][feature]:        
                sum_prob+=probabilities[person][feature][item]
            
            norm_factor = 1/sum_prob    
            for item in probabilities[person][feature]:
                probabilities[person][feature][item]=norm_factor * probabilities[person][feature][item]  

if __name__ == "__main__":
    main()
