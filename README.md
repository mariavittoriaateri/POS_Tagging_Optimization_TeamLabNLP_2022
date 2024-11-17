# NLP Team Lab Summer Semester 2022 - POS Tagging Task - Team 8 (Maria Vittoria Ateri, Malak Rassem)

## Repo Structure
There are 4 main files. 
1. [HMM.py](HMM.py) - Runnable with Hidden Markov Model method.
2. [CRF.py](CRF.py) - Runnable with Conditional Random Field method.
3. [Evaluation.py](Evaluation.py) -Support file. Contains evaluation code. This is automatically called from the HMM and CRF files. 
4. [Utils.py](Utils.py) -Support file. Contains some methods for setup or methods that do not particularly fit in the other files (e.g. CSV creation)

## How To Run
For HMM method: <br />
  * Simply run [HMM.py](HMM.py) file. <br />
  * Dialog boxes will prompt you to choose the training and test file. <br />
  * When the code is done, you will be asked if you want to save a CSV with the results or not. You will then be prompted to name the file and select its save location.
  <br />
  
For CRF method: <br />
  * Simply run [CRF.py](CRF.py) file. <br />
  * Dialog boxes will prompt you to choose the training and test file and the location where you would like to save the results (37 .csv files). <br />
  * When the code is done, you will be asked if you want to have a merged CSV file with all the results. This does not delete the other files. It is just for convenience. <br />

