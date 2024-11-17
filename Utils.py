import numpy as np
import glob
import os
import re
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename,asksaveasfilename
from tkinter.messagebox import askyesno

training=standard=unique_tags=tag_counts=unique_words=[]
deleted_tags=['XX','AFX','NFP'] #Irrelevant tags to be deleted

root=Tk()
root.attributes("-topmost", True) #To focus dialog
root.withdraw() # no full GUI needed

def getData():
    global training, standard,training,standard
    training_file_path = askopenfilename(parent=root,filetypes=[("Col Files",'.col')], title='Select training data') # Get path to selected training file
    testing_file_path = askopenfilename(parent=root,filetypes=[("Col Files",'.col')], title='Select testing data') #Get path to selected testing file   
    training=np.loadtxt(training_file_path,dtype='str') #loading training data
    standard=np.loadtxt(testing_file_path,dtype='str') #loading gold standard of predicted data
    return training,standard

def setUp():
    global unique_tags,tag_counts,unique_words
    unique_tags, tag_counts= np.unique(training[:,1],return_counts=True) #Getting all possible unique values of the POS tag  and their respective occurrences
    unique_words= np.unique(np.char.lower(training[:,0]))#Getting all possible unique words
    return unique_tags, tag_counts,unique_words

def csvMaker(df):
    csv = askyesno(parent=root, title='CSV Creation', message='Do you want a .csv file with the results?')
    if csv:
        path = asksaveasfilename(parent=root,defaultextension=".csv", filetypes=[("Comma-Seperated File",".csv")])
        path = path if ".csv" in path else path + ".csv"
        df.to_csv(path,index=False)

#Reference for Human Sorting https://nedbatchelder.com/blog/200712/human_sorting.html
def atof(text): #Helper Method for Merging CSVs
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text): #Helper Method for Merging CSVs For Human Sorting.
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def mergeCSV(path):
    merge = askyesno(parent=root,title='CSV Merging', message='Do you want to merge the results into one CSV file?')
    if merge:
        flist=glob.glob(os.path.join(path, 'ForwardSelectionResultsIter*.csv')) #Get all files that start with this.
        files=[]
        flist.sort(key=natural_keys) #To avoid having file 10 after file 1 instead of 2. 
        for f in flist:
            name=f.split('ForwardSelectionResultsIter')[1][:-4]           
            files.append(pd.DataFrame(columns=[name])) #Add a column with the file number before results for better readability and to ensure it is the correct file.
            files.append(pd.read_csv(f))
        df=pd.concat(files, axis=1)
        df.to_csv(os.path.join(path,"ForwardSelectionResultsMerged.csv"), index=False)

def getSaveLocation():
    return askdirectory(parent=root,initialdir="",title="Where do you want to save the results?")

