import os
import numpy as np
import pandas
import string
import sklearn_crfsuite
from Utils import getSaveLocation, getData, mergeCSV, setUp

#Initializing training and test data.
training,test=getData()
save_path=getSaveLocation()
setUp()
print("Hold tight it is gonna take a while :)- Almost 24 hours")
training_sents= np.split(training, np.where(training[:, 0]==".")[0][1:] +1)
test_sents= np.split(test, np.where(test[:, 0]==".")[0][1:] +1)

turned_off_features=[]
f_count=0

#Define FW set
fw= training[training[:,1]=='FW'][:,0]
fw = set([w.lower() for w in fw])


def endsWithS(word):
       return word[-1:]=='s'

def endsWithST(word):
       return word[-2:]=='st'

def endsWithER(word):
       return word[-2:]=='er'

def isCardNum(word):
    if word.isdigit():
        return True
    else:
        cardnumbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion']
        word = word.lower()
        return word in cardnumbers

def isForeign(word): 
    return (word.lower() in fw)
    
def getFeatures(sent,word,i):
    features={}
    #Features related to current word
    if f_count>-1 and 0 not in turned_off_features: 
        features['word.lower()']=word.lower()
    if f_count>0 and 1 not in turned_off_features:
        features['startUpper']=word[0].isupper()
    if f_count>1 and 2 not in turned_off_features:
        features['endsWithST']= endsWithST(word)
    if f_count>2 and 3 not in turned_off_features:
        features['endsWithER']=endsWithER(word)    
    if f_count>3 and 4 not in turned_off_features:
        features['endsWithS']=endsWithS(word)
    if f_count>4 and 5 not in turned_off_features:
        features['word[-2:]']=word[-2:]   
    if f_count>5 and 6 not in turned_off_features:
        features['word.iscard()']= isCardNum(word)
    if f_count>6 and 7 not in turned_off_features:
        features['word.isdigit()']= word.isdigit()
    if f_count>7 and 8 not in turned_off_features:
        features['word.isForeign()']= isForeign(word)
    if f_count>8 and 9 not in turned_off_features:
        features['word.ispunctuation']=(word in string.punctuation)
    if f_count>9 and 10 not in turned_off_features:
        features['len(word)']=len(word)  
    if f_count>10 and 11 not in turned_off_features:
        if i>0:
            features['BOS'] = False
        else: 
            features['BOS'] = True
    if f_count>11 and 12 not in turned_off_features:
        if i < len(sent)-1:
            features['EOS'] = False
        else:
            features['EOS'] = True
    #One Word Before
    if f_count>12 and 13 not in turned_off_features:
        if i>0:
            word1 = sent[i-1][0]
            features['-1:word']= word1.lower()
        else: 
            features['-1:word']= ""
    if f_count>13 and 14 not in turned_off_features:
        if i>0:
            word1 = sent[i-1][0]
            features['-1wordupper']=word1[0].isupper()
        else: 
            features['-1wordupper']= ""
    if f_count>14 and 15 not in turned_off_features:
        if i>0:
            word1 = sent[i-1][0]
            features['-1:word.ispunctuation']=(word1 in string.punctuation)
        else: 
            features['-1:word.ispunctuation']= ""
    if f_count>15 and 16 not in turned_off_features:
        if i>0:
            word1 = sent[i-1][0]
            features['-1:word.iscard()']=isCardNum(word1)
        else: 
            features['-1:word.iscard()']= ""        
    if f_count>16 and 17 not in turned_off_features:
        if i>0:
            word1 = sent[i-1][0]
            features['isForeign(word-1)']=isForeign(word1)
        else: 
            features['isForeign(word=1)']= ""  
    if f_count>17 and 18 not in turned_off_features:
        if i>0:
            word1 = sent[i-1][0]
            features['-1:word[-2:]']=word1[-2:]
        else: 
            features['-1:word[-2:]']= ""
    #One Word After
    if f_count>18 and 19 not in turned_off_features:
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            features['+1:word']= word1.lower()
        else: 
            features['+1:word']= ""
    if f_count>19 and 20 not in turned_off_features:
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            features['+1wordupper']=word1[0].isupper()
        else: 
            features['+1wordupper']= ""
    if f_count>20 and 21 not in turned_off_features:
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            features['+1:word.ispunctuation']=(word1 in string.punctuation)
        else: 
            features['+1:word.ispunctuation']= ""
    if f_count>21 and 22 not in turned_off_features:
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            features['+1:word.iscard()']=isCardNum(word1)
        else: 
            features['+1:word.iscard()']= ""        
    if f_count>22 and 23 not in turned_off_features:
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            features['isForeign(word+1)']=isForeign(word1)
        else: 
            features['isForeign(word+1)']= ""  
    if f_count>23 and 24 not in turned_off_features:
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            features['+1:word[-2:]']=word1[-2:]
        else: 
            features['+1:word[-2:]']= ""
    #Two Words Before
    if f_count>24 and 25 not in turned_off_features:
        if i>1:
            word2 = sent[i-2][0]
            features['-2:word']= word2.lower()
        else: 
            features['-2:word']= ""
    if f_count>25 and 26 not in turned_off_features:
        if i>1:
            word2 = sent[i-2][0]
            features['-2wordupper']=word2[0].isupper()
        else: 
            features['-2wordupper']= ""
    if f_count>26 and 27 not in turned_off_features:
        if i>1:
            word2 = sent[i-2][0]
            features['-2:word.ispunctuation']=(word2 in string.punctuation)
        else: 
            features['-2:word.ispunctuation']= ""
    if f_count>27 and 28 not in turned_off_features:
        if i>1:
            word2 = sent[i-2][0]
            features['-2:word.iscard()']=isCardNum(word2)
        else: 
            features['-2:word.iscard()']= ""        
    if f_count>28 and 29 not in turned_off_features:
        if i>1:
            word2 = sent[i-2][0]
            features['isForeign(word-2)']=isForeign(word2)
        else: 
            features['isForeign(word-2)']= ""  
    if f_count>29 and 30 not in turned_off_features:
        if i>1:
            word2 = sent[i-2][0]
            features['-2:word[-2:]']=word2[-2:]
        else: 
            features['-2:word[-2:]']= ""
    #Two Words After
    if f_count>30 and 31 not in turned_off_features:
        if i < len(sent)-2:
            word2 = sent[i+2][0]
            features['+2:word']= word2.lower()
        else: 
            features['+2:word']= ""
    if f_count>31 and 32 not in turned_off_features:
        if i < len(sent)-2:
            word2 = sent[i+2][0]
            features['+2wordupper']=word2[0].isupper()
        else: 
            features['+2wordupper']= ""
    if f_count>32 and 33 not in turned_off_features:
        if i < len(sent)-2:
            word2 = sent[i+2][0]
            features['+2:word.ispunctuation']=(word2 in string.punctuation)
        else: 
            features['+2:word.ispunctuation']= ""
    if f_count>33 and 34 not in turned_off_features:
        if i < len(sent)-2:
            word2 = sent[i+2][0]
            features['+2:word.iscard()']=isCardNum(word2)
        else: 
            features['+2:word.iscard()']= ""        
    if f_count>34 and 35 not in turned_off_features:
        if i < len(sent)-2:
            word2 = sent[i+2][0]
            features['isForeign(word+2)']=isForeign(word2)
        else: 
            features['isForeign(word+2)']= ""  
    if f_count>35 and 36 not in turned_off_features:
        if i < len(sent)-2:
            word2 = sent[i+2][0]
            features['+2:word[-2:]']=word2[-2:]
        else: 
            features['+2:word[-2:]']= ""
    return features

def sent2features(sent):
    #Adds features for every word in a given sentence 
    sentence_features=[]
    for i,word_tag in enumerate(sent):
        word = word_tag[0]
        word_features=getFeatures(sent,word,i)
        sentence_features.append(word_features)
    return sentence_features

def sent2labels(sent):
    return [word[1] for word in sent]

ytrain = [sent2labels(s) for s in training_sents]

#Define CRF model with standard settings
crf = sklearn_crfsuite.CRF(
    algorithm = 'lbfgs',
    c1 = 0.25,
    c2 = 0.3,
    max_iterations = 100,
    all_possible_states= True
)

from Evaluation import Evaluate
previous_macro=0

while f_count<=36: #Go through the 37 features.
    Xtrain = [sent2features(s) for s in training_sents]
    Xtest = [sent2features(s) for s in test_sents]
    try:
        crf.fit(Xtrain, ytrain)
    except AttributeError:
        pass
    #Get Predictions and evaluate iteration
    predictions = [val for sublist in crf.predict(Xtest) for val in sublist]
    df,macro,_=Evaluate(predictions,test)
    path=os.path.join(save_path,"ForwardSelectionResultsIter"+str(f_count)+".csv")
    df.to_csv(path,index=False)
    #If this iteration is worse than last, skip this feature in the next iteration.
    if macro<previous_macro:
        turned_off_features.append(f_count)
    else:
        previous_macro=macro
    print("I finished and saved iteration "+str(f_count)+"! :)")
    f_count+=1

print("I am finally done! :) Check out the files that I saved")
print("Bad features that were eliminated are: ",turned_off_features)
mergeCSV(save_path)



