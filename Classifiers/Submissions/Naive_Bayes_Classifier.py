## extracting a cluster
def getOneLabel(df,label,predLabel):
  tempDf = pd.read_csv('Datasets/stolenCars.csv')
  x = [df.loc[i][label] in predLabel for i in df.index]
  count = 0
  ind = []
  for val in x:
    if not val:
      ind.append(count)
    count += 1
  
  #print(ind)
  tempDf.drop(ind, inplace=True)
  return tempDf

## Naive Bayes Classifier
def predict(df,labels, given, predCol, describe = False):
  given = groupCol(dataCol=given)
  
  finalPrediction = naiveBayesianPredictor(df, labels, given, predVar)
  
  if finalPrediction == -1:
    print("Tie for the given e-mail!! Get a human to check for spam!")
  else:
    res = ' ' if bool(int(finalPrediction[0])) else ' NOT '
    print("\nGiven e-mail is{}Spam! \nLabel Value: {}".format(res,finalPrediction[0]))
  if describe:
    print("\nHere 1 indicates the probability of being a Spam E-Mail!")
    print("Probability Table:")
    return pd.DataFrame([finalPrediction[1]])
