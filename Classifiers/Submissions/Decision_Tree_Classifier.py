"""decisionTreeCore.ipynb

This python script is converted from the python notebook and automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/ravi-prakash1907/Machine-Learning-for-Cyber-Security/blob/background/Classifiers/decisionTreeCore.ipynb

# Working with Decision Tree
"""

from math import log2
import pandas as pd

"""## Data Collection"""

import requests

def downloadCSV(fileURL, saveAs='downloaded.csv'):
  req = requests.get(fileURL)
  fileURLContent = req.content
  csv_file = open(saveAs, 'wb')
  
  csv_file.write(fileURLContent)
  csv_file.close()

#get data
downloadCSV("https://raw.githubusercontent.com/ravi-prakash1907/Machine-Learning-for-Cyber-Security/main/Datasets/decisionTreeSample.csv?token=AJGAAOHPMO2B2C6UPVPQ5I3ARUNLI","data.csv")

"""## Algo Requirements

### Entropy
"""

def getEntropy(df, colPredict):
  labels = df[colPredict].unique()
  total = len(df)
  entropy = 0
  for l in labels:
    tempDF = df[df[colPredict] == l]
    count = len(tempDF)
    Pi = count/total
    entropy += -Pi * log2(Pi)
  return entropy

# getEntropy(df,'Class')

"""### Gini Index"""

def getGiniIndex(df, colPredict):
  labels = df[colPredict].unique()
  total = len(df)
  partialIndex = 0
  for l in labels:
    tempDF = df[df[colPredict] == l]
    count = len(tempDF)
    Pi = count/total
    partialIndex += Pi**2
  giniIndex = 1-partialIndex
  return giniIndex

# getGiniIndex(df,'Class')

def getAttrGiniIndex(df, attr, colPredict):
  target = df[attr].unique()
  partialGiniIndex = []

  ## info gain for every val in label
  for t in target:
    tempDF = df[df[attr] == t]
    tempGiniIndex = getGiniIndex(tempDF,colPredict)
    partialGiniIndex.append(tempGiniIndex)
  
  ## final gini index for attr
  finalGiniIndex = getGiniIndex(df,colPredict) - sum(partialGiniIndex)
  return finalGiniIndex

"""### Information Gain"""

def getAttrEntropy(df, attr, colPredict):
  target = df[attr].unique()
  partialEntropies = []

  ## info gain for every val in label
  for t in target:
    tempDF = df[df[attr] == t]
    tempEntropy = getEntropy(tempDF,colPredict)
    tempEntropy *= len(tempDF)/len(df)
    partialEntropies.append(tempEntropy)

  return partialEntropies

# getAttrEntropy(df,'Age','Class')

def getInfoGain(df,attr,colPredict):
  avgEntropies = getAttrEntropy(df, attr, colPredict)
  infoGain = getEntropy(df,colPredict) - sum(avgEntropies)
  return infoGain

# getInfoGain(df,'Age','Class')

"""### Gain Ratio"""

def getGainRatio(df, attr, colPredict):
  infoGain = getInfoGain(df, attr, colPredict)
  entropy = getEntropy(df,colPredict)
  gainRatio = infoGain/entropy
  return gainRatio

# getGainRatio(df,'Age','Class')


## Algo. for selecting root _(in decision tree)_

#### loading data
df = pd.read_csv('data.csv')
df.head()

df.shape

## constats
colToPredict = 'Class'
comparisionMat = pd.DataFrame(columns=['Algorithm','Root Attribute'])

def addRow(df, algo, root):
    #create rows for comparision
    thisRow = {"Algorithm":algo,
               "Root Attribute":root}
    thisRow = pd.Series(thisRow)
    df = df.append(thisRow,ignore_index=True)
    
    return df

"""### ID3"""

def id3(df, predictionCol):
  attributes = list(df.columns)
  attributes.remove(predictionCol)
  attrCount = len(attributes)

  infoGains = list(map(getInfoGain, [df]*attrCount, attributes, [predictionCol]*attrCount))
  rootAttr = df.columns[infoGains.index(max(infoGains))]

  return rootAttr

root = id3(df,colToPredict)
comparisionMat = addRow(comparisionMat,'ID3',root)

"""### CART"""

def cart(df, predictionCol):
  attributes = list(df.columns)
  attributes.remove(predictionCol)
  attrCount = len(attributes)

  giniIndex = list(map(getAttrGiniIndex, [df]*attrCount, attributes, [predictionCol]*attrCount))
  rootAttr = df.columns[giniIndex.index(max(giniIndex))]

  return rootAttr

root = cart(df,colToPredict)
comparisionMat = addRow(comparisionMat,'CART',root)

"""### C4.5"""

def c4dot5(df, predictionCol):
  attributes = list(df.columns)
  attributes.remove(predictionCol)
  attrCount = len(attributes)

  gainRatio = list(map(getGainRatio, [df]*attrCount, attributes, ['Class']*attrCount))
  rootAttr = df.columns[gainRatio.index(max(gainRatio))]

  return rootAttr

root = c4dot5(df,colToPredict)
comparisionMat = addRow(comparisionMat,'C4.5',root)

## Comparision
print("Given dataset (sample):\n")
print(df.head())

print("Comparision matrix for root selection for decision tree:\n")
print(comparisionMat)
