import numpy as np
from Utils import csvMaker, getData, setUp

training,standard=getData() 
print("Please wait, it's gonna take a while :)")
unique_tags, counts, unique_words= setUp()
unique_words=dict(zip(unique_words,np.arange(0,len(unique_words)))) #formatting unique words as a dictionary alongside their index for quicker index retrieval (O(1))

transitions={key: list(np.zeros((len(unique_tags)))) for key in unique_tags} 
emissions=np.zeros((len(unique_tags),len(unique_words))) 

#Filling in the Transition and Emission matrices.
previous_tag="." #Initializing previous tag.
for row in training:
    tag=row[1]
    word=str(np.char.lower(row[0]))
    index_word=unique_words[word]
    index_tag=unique_tags.tolist().index(tag) #Possible here due to small number of tags.
    emissions[index_tag][index_word]+=1
    transitions[previous_tag][index_tag]+=1
    previous_tag=tag
emissions= dict(zip(unique_tags,emissions))

#Formatting the emission and transition values as probabilities.
for context_word in transitions:
    transitions[context_word]=np.add(transitions[context_word],1)/(np.sum(transitions[context_word])+len(unique_words)) #Add-one smoothing.
    with np.errstate(divide='ignore'):
        emissions[context_word]=emissions[context_word]/np.sum(emissions[context_word])
emissions=np.array(list(emissions.values())) #No longer need it as a dictionary.

# VITERBI ALGORITHM
# This article really helps with it
# https://medium.com/analytics-vidhya/parts-of-speech-pos-and-viterbi-algorithm-3a5d54dfb346

accumulator = np.zeros((len(unique_tags), standard.shape[0])) #Initialize accumulator 
accumulator[:,0] = np.multiply(transitions["."], emissions[:, unique_words[str(np.char.lower(standard[:,0][0]))]]) #start state transition * emissions of the first word in test data

transitions_arr = np.array(list(transitions.values())) #for easier retrieval 

for j in range(1,accumulator.shape[1]):
    current_word = str(np.char.lower(standard[:, 0][j]))
    previous_accumulation = np.nanmax(accumulator[:, j - 1])
    maximum_previous_tag=np.nanargmax(accumulator[:,j-1])
    for i in range(accumulator.shape[0]):
        current_transition=transitions_arr[:,i]
        try:
            current_emission= emissions[:, unique_words[str(np.char.lower(standard[:,0][j]))]][i] #How likely is it for current word to be this tag.
            if current_emission==0:
                continue #Will keep it zero as it should be. 
        except KeyError:  #If the word doesn't exist in the training data we do not have its emissions.
            current_emission=(1/48)  #Assign equal probabilities to all tags.
        with np.errstate(divide='ignore'):
            accumulator[:, j][i] = np.exp(np.exp(np.log(previous_accumulation) + np.log(current_transition[maximum_previous_tag]) + np.log(current_emission)))
            # storing exp instead of repeatedly taking log reference: 
            # https://journals.sagepub.com/doi/pdf/10.1177/1550147718772541

# BACKWARDS PASS
prediction=[]
for j in range(accumulator.shape[1]-1,-1,-1):
    prediction.insert(0,unique_tags[np.argmax(accumulator[:,j])]) #Get label of the maximum in the accumulator.

#Evaluate Prediction
from Evaluation import Evaluate
results_df,_,_=Evaluate(prediction,standard)
print(results_df)
csvMaker(results_df)