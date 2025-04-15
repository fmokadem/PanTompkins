from sklearn.naive_bayes gaussnb;

def classify(data):
  nb = gaussnb()
  return nb.fit(data[0], data[1]).predict(data[0])
