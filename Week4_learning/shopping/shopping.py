import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - 0-Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - 5-ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - 10-Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - 15-VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    #create the two lists
    evidence = []
    labels =[]
    
    # Read data in from file
    with open(filename) as f:
       reader = csv.reader(f)
       next(reader)
       #fill each list
       for row in reader:
          temp_evi = []
          months=['Jan','Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec']
          temp_evi.append(int(row[0]))
          temp_evi.append(float(row[1]))
          temp_evi.append(int(row[2]))
          temp_evi.append(float(row[3]))
          temp_evi.append(int(row[4]))
          temp_evi.append(float(row[5]))
          temp_evi.append(float(row[6]))
          temp_evi.append(float(row[7]))
          temp_evi.append(float(row[8]))
          temp_evi.append(float(row[9]))
          num1=[x for x in range(12) if row[10]==months[x]]
          temp_evi.append(num1[0])
          temp_evi.append(int(row[11]))
          temp_evi.append(int(row[12]))
          temp_evi.append(int(row[13]))
          temp_evi.append(int(row[14]))
          num2=[1 if row[15]=='Returning_Visitor' else 0]
          temp_evi.append(num2[0])
          num3=[1 if row[16]=='TRUE' else 0]
          temp_evi.append(num3[0])
          evidence.append(temp_evi)
   
          labels.append(1 if row[-1]=="TRUE" else 0)     
    
    #return tuple
    return(evidence,labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    classif= KNeighborsClassifier(n_neighbors=1)
    classif.fit(evidence,labels)

    return classif
    
def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    #initiate the variables
    sens,spec=0,0
    
    #count pos and neg values 
    all_pos = sum(labels)
    all_neg = len(labels) - all_pos
    
    #count the correctly predicted pos and neg values by looping over the combi's
    #initiate the variables
    sens,spec=0,0
        
    for lab,pred in zip(labels,predictions):      
        if lab==1 and pred==1:
           sens+=1
        elif lab==0 and pred==0:
           spec+=1
        else:
           continue

    #divide to find sensitivity and specificity         
    sensi = sens/all_pos
    speci = spec/all_neg
    
    return(sensi,speci)


if __name__ == "__main__":
    main()
