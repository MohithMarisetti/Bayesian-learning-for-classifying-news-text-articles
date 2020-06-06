import os
import math
import random

# Tweakable variables

"""test_size: This defines the percentage of data which needs to be used for testing the classifier. Remaining data will 
be used for training. """
test_size = 0.5
seed_value = 4     # This is used to shuffle the data
# These Thresholds are not currently being used. This was done as a part of tweaking the performance.
deleteThresholdUpperBound = 1000   # This defines the upper bound for the frequeny of a word. For example: If a words appears more than 1000 times, we remove it.
deleteThresholdLowerBound = 0      # This defines the lower bound for the frequency of a word. For example: If a word appears less than 2 times then remove it. 



# Global variables to be used in the code
totalNumberOfDocuments = 0          # Total number of documents used for training the classifier.
count = 0 
path = './20_newsgroups/'           # The path of data.
file_count = 0                      # Number of files read in each class.
numberOfDocumentsInLabel = {}       # This stores number of documents in each class label that was read during training. 
uniqueWordsInAllLabels = []         # This stores the unique words in all the class labels
numberOfWordsInEachClass = {}       # This dictionary stores the number of words each class contains. Note: This variable is printed in the output. So that you can have a look at it.  
stopwords = []                      # Stores the list of stop words in the stopwords.txt file
words = []                          # To store each word 
sw = 0                              # Stop words counter variable (Used this to see how many stop words occured in the documents.)

# Testing variables. (The variables which I used during testing for correctness.)
notword = 0                         # Non english words counter. 
files_in_train = []                 # files in the train
files_in_test = []                  # files in the test


# Reading the stop words from the file and storing it into "stopwords" list
with open("stopwords.txt","r") as swfile:
    for line in swfile.readlines():
        word = line.split()[0]
        stopwords.append(str(word))


# This function will clean the word and remove all the special characters from it.
def preprocessWord(eachword):
    eachword = eachword.lower()
    if ("," in eachword):
        eachword = eachword.replace(",","")
    if ("." in eachword):
        eachword = eachword.replace(".","")
    if (">" in eachword):
        eachword = eachword.replace(">","")
    if ('"' in eachword):
        eachword = eachword.replace('"',"")
    if ("'" in eachword):
        eachword = eachword.replace("'","")
    if ("?" in eachword):
        eachword = eachword.replace("?","")
    if ("(" in eachword):
        eachword = eachword.replace("(","")
    if (")" in eachword):
        eachword = eachword.replace(")","")
    if ("[" in eachword):
        eachword = eachword.replace("[","")
    if ("]" in eachword):
        eachword = eachword.replace("]","")
    if ("\\" in eachword):
        eachword = eachword.replace("\\","")
    if ("/" in eachword):
        eachword = eachword.replace("/","")
    if ("|" in eachword):
        eachword = eachword.replace("|","")
    if ("<" in eachword):
        eachword = eachword.replace("<","")
    if ("^" in eachword):
        eachword = eachword.replace("^","")
    if ("&" in eachword):
        eachword = eachword.replace("&","")
    if ("*" in eachword):
        eachword = eachword.replace("*","")
    if(":" in eachword):
        eachword = eachword.replace(":","")
    if("-" in eachword):
        eachword = eachword.replace("-","") 
    if ("_" in eachword):
        eachword = eachword.replace("_","")
    if ("=" in eachword):
        eachword = eachword.replace("=","")
    if("+" in eachword):
        eachword = eachword.replace("+","") 
    if ("*" in eachword):
        eachword = eachword.replace("*","")
    if (";" in eachword):
        eachword = eachword.replace(";","")
    if ("@" in eachword):
        eachword = eachword.replace("@","")
    if ("!" in eachword):
        eachword = eachword.replace("!","")
    if ("~" in eachword):
        eachword = eachword.replace("~","")
    if ("`" in eachword):
        eachword = eachword.replace("`","")
    if ("#" in eachword):
        eachword = eachword.replace("#","")
    if ("$" in eachword):
        eachword = eachword.replace("$","")
    if ("%" in eachword):
        eachword = eachword.replace("%","")
    if ("{" in eachword):
        eachword = eachword.replace("{","")
    if ("}" in eachword):
        eachword = eachword.replace("}","")
    return eachword

def extractDictionary(root,files):
      
    global totalNumberOfDocuments
    global file_count
    global numberOfDocumentsInLabel
    global numberOfWordsInEachClass
    global sw           
    global notword 
    
    priorProbabilityDict = {}       # This is used to store class label word frequency counts.

    for filename in files:    
        # Logic to read only the 50% of the train data.
        file_count+=1
        if (file_count >= (len(files)*(1-test_size))):
            break

        f = open(root+"/"+filename,'r')
        # files_in_train.append(filename)
        
        for eachline in f.readlines():
            # Removing the header information
            if ("Path" in eachline or "Xref" in eachline or "From" in eachline or "lines" in eachline or "[...]" in eachline):
                continue
            
            for eachWord in eachline.split():

                # Words with greater length than 15 and shorter than length 2 usually would be incorrect words. So, I removed them
                if ((len(eachWord) >= 15) or (len(eachWord) <= 2)):
                    continue
                eachWord = preprocessWord(eachWord)      # Preprocess this word
                if (eachWord in stopwords):     # Removing stop words
                    sw+=1
                    continue
                if(eachWord.isdigit() ):            # Removing digits
                    continue
                if ((len(eachWord) >= 15) or (len(eachWord) <= 2)): # Words with greater length than 15 and shorter than length 2 usually would be incorrect words. So, I removed them
                    continue

                # Creating dictionary logic
                if (priorProbabilityDict.get(eachWord) == None):
                    priorProbabilityDict[eachWord] = 1
                else:
                    priorProbabilityDict[eachWord] += 1
            
    totalNumberOfDocuments+=file_count          # Counting the total documents read
    numberOfDocumentsInLabel[root] = file_count # Storing the number of documents in each class label
    file_count = 0                      # Reset the file count
    return priorProbabilityDict
    
if __name__ == "__main__":
            
    probabilityDictionaryCollection = {}    

    for root, dicts, files in os.walk(path): 
        if (count !=3):
            count+=1
        if (count==1):
            continue
        # Shuffling the data
        random.seed(seed_value)
        random.shuffle(files)

        probabilityDictionaryCollection[root] =  extractDictionary(root,files)

    # print("Number of stop words encountered = {}".format(sw))

    # print("The entire dictionary is ")
    # print(probabilityDictionaryCollection)


    #Approach-2: Delete the intersection of words in all the lists
    temp_list = []
    for key, val in probabilityDictionaryCollection.items():
        temp_list.append(set(list(val.keys())))
    common_words = temp_list[0]
    for eachSet in temp_list:
        common_words = common_words.intersection(eachSet)

    temp_dictionary = probabilityDictionaryCollection.copy()
    common_words_delete_count = 0
    for it in common_words:
        for key, val in temp_dictionary.items():
            del probabilityDictionaryCollection[key][it]
            common_words_delete_count+=1
            
    print("Number of common words deleted = {}".format(common_words_delete_count))


    """
    # Approach-1: Delete the top frequent words.
    totalCount = 0
    tempDict = probabilityDictionaryCollection.copy()
    deletedCount = 0


    for key, val in tempDict.items():
        
            for key2, val2 in val.copy().items():
                if(val2 >= deleteThresholdUpperBound or val2 <= deleteThresholdLowerBound):
                    # if(probabilityDictionaryCollection[key][index].get(key2) !=None):
                    #     del probabilityDictionaryCollection[key][index][key2]   
                    #     deletedCount+=1
                    del probabilityDictionaryCollection[key][key2]   
                    deletedCount+=1


    print("Deleted {} top frequent items".format(deletedCount))
    """


    # Finding the number of words in each class and also the total unique words in all the class labels
    words = []
    for index, (key, val) in enumerate(probabilityDictionaryCollection.items()):
        numberOfWords = 0   
        words += list(val.keys())
        numberOfWords += len(val.keys())
        numberOfWordsInEachClass[key] = numberOfWords
    
    uniqueWordsInAllLabels = set(words)

    print("Number of words in each class = {}".format(numberOfWordsInEachClass))
    print("Number of Unique words in all labels = {}".format(len(uniqueWordsInAllLabels)))

    # print("Unique words in all labels are:")
    # print(uniqueWordsInAllLabels)



    # Checking the probability of this file belonging to each of the categories. Starting with p(alt.atheism|graphics) >NOTE: Graphics is the name of the file.
    
    correct = 0 # Correct predictions counter variable
    incorrect = 0   # InCorrect predictions counter variable

    path = './20_newsgroups/'

    for count, (root, dicts, files) in enumerate(os.walk(path)): 
        if (count !=3):
            count+=1
        if (count==1):
            continue

        random.seed(seed_value)
        random.shuffle(files)

        for file_count, filename in enumerate(files):
            file_count+=1
            # Reading only the test files
            if (file_count <= (len(files)*(1-test_size))):    
                continue
            
            final_values_list = []
            for eachKey in probabilityDictionaryCollection.keys():
                final_value = 0
                f = open(root+"/"+filename,"r")
                # files_in_test.append(filename)    
                for eachline in f.readlines():

                    # Same preprocessing logic
                    if ("Path" in eachline or "Xref" in eachline or "From" in eachline or "lines" in eachline or "[...]" in eachline):
                        continue
                        
                    for eachword in eachline.split():
                        if ((len(eachword) >= 15) or (len(eachword) <= 2)):
                            continue
                        eachword = preprocessWord(eachword)
                        if (eachword in stopwords):
                            continue

                        if(eachword.isdigit()):
                            continue
                        
                        if ((len(eachword) >= 15) or (len(eachword) <= 2)):
                            continue



                        wordCount = 0
                        # Calculating the word count and then using it for the equation in Naive Bayes.
                        if(probabilityDictionaryCollection[eachKey].get(eachword)!=None):
                            wordCount += probabilityDictionaryCollection[eachKey].get(eachword)
                        
                        if (wordCount == 0):        # If that term doesnt exist in the document dont consider that term
                            continue
                        # Naive Bayes equation as shown in the report. (using log to make the values smaller)
                        final_value= final_value +  math.log( wordCount + 1 )      
                        # final_value= final_value * (( wordCount + 1)/(numberOfWordsInEachClass[eachKey]+len(uniqueWordsInAllLabels)))  # This equation has almost all times a denominator very huge. So the value becomes 0 eventually
                prior = (numberOfDocumentsInLabel[eachKey]/totalNumberOfDocuments)   
                try:
                    final_value = final_value + math.log(prior)
                except:
                    pass
                final_values_list.append(final_value)
                
            try:
                # Predicting as the class label which has the max value.
                prediction = list(probabilityDictionaryCollection.keys())[final_values_list.index(max(final_values_list))]
            except:
                pass
            
            target = root   # Simply storing the directory path as a way to compare predictions of class labels.
            if (prediction.lower() == target.lower()):
                # Uncomment the below 2 lines if you wish to see what predictions were correct.
                # print("Prediction correct.", end = " ")
                # print("{} = {}".format(prediction.lower(), target.lower()))
                correct+=1

            else:
                # Uncomment the below 2 lines if you wish to see what predictions were incorrect.
                # print("incorrect  prediction.", end= '  ')
                # print("predicted != Target i.e., {} != {}".format(prediction.lower(), target.lower()))
                incorrect+=1

    print("\nAccuracy = {}%".format(100*(correct/(correct+incorrect))))



