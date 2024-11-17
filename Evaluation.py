import numpy as np
import pandas as pd
from Utils import unique_tags as labels, tag_counts as label_counts, deleted_tags

# In order to implement the evaluation method of Fscore, we used this website as reference:
# https://www.baeldung.com/cs/multi-class-f1-score

def CompareLabels(looper,checker): 
    #In order to get fp, we iterate over what we predicted and compare it with the standard value. looper=predicted , checker=standard
    #In order to get fn, we iterate over the correct label and compare it with our predicted value. looper=standard , checker=predicted
    #Tp can be evaluated both ways
    
    tp=dict.fromkeys(labels, 0)
    f_=dict.fromkeys(labels, 0)
    for i,x in enumerate(looper):
        #If same label increment tp of the particular class.
        if x[1]==(checker[i][1]):
            tp[x[1]]=tp.get(x[1])+1
        else:
            # NOT the same increment the fp/fn of the predicted class.
            f_[x[1]]=f_.get(x[1])+1
    return tp,f_

def Precision(predicted,standard): #tp/(tp+fp)
    tp,fp=CompareLabels(predicted,standard)
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.divide(np.array(list(tp.values())),np.array(list(tp.values()))+np.array(list(fp.values())))   

def Recall(predicted,standard): #tp/(tp+fn)
    tp,fn=CompareLabels(standard,predicted)
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.nan_to_num(np.divide(np.array(list(tp.values())),np.array(list(tp.values()))+np.array(list(fn.values())))) 

def FScore(predicted,standard): #(2PR)/(P+R)
    #The fscore is a list of 48 members. One score for each class.
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.nan_to_num((2*Precision(predicted,standard)*Recall(predicted,standard))/(Precision(predicted,standard)+Recall(predicted,standard))) 

def Evaluate(predicted,standard):
    predicted=np.insert(standard, 1, predicted, axis=1)[:,:2] #Adding the column of the results and removing the real labels

    #Get Scores
    recall=Recall(predicted,standard)
    precision=Precision(predicted,standard)
    fscore=FScore(predicted,standard)

    eval=np.stack((precision,recall,fscore),axis=-1)
    weighted_eval= (label_counts*eval.T).T 

    #Remove rows that correspond to deleted tags.
    eval=np.delete(eval,np.where(np.isin(labels,deleted_tags)),axis=0)
    weighted_eval=np.delete(weighted_eval,np.where(np.isin(labels,deleted_tags)),axis=0)
    updated_label_counts = np.delete(label_counts,np.where(np.isin(labels,deleted_tags)))
    updated_labels=np.setdiff1d(labels,deleted_tags)

    #Add Macro and Weighted Average
    macro=np.nanmean(eval,axis=0)
    weighted_average=np.nansum(weighted_eval,axis=0)/np.sum(updated_label_counts)
    eval=np.append(eval,[macro,weighted_average],axis=0)

    #Create dataframe
    results_df=pd.DataFrame(eval,columns=["Precision","Recall","F1-Score"])
    updated_labels=np.append(updated_labels,["macro","weighted_average"])
    results_df.insert(0,'Labels',updated_labels)

    #Returns whole dataframe, macro f1score, weighted average
    return results_df,results_df.at[len(results_df)-2,"F1-Score"],results_df.at[len(results_df)-1,"F1-Score"]

