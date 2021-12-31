# Import the  required libraries 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from collections import defaultdict
from math import sqrt
from collections import Counter
from sklearn.model_selection  import train_test_split

# Data manipulation in simple words , Convert the dataframe(or the csv) to a list of dictionaries , classes as key , it will be easier for programming .
iris = pd.read_csv('/content/Iris - all-numbers.csv') # make sure to convert categoraical data to numerical , the csv file is in my repo in case if u need .
print(iris.head(4))
iris = iris.values.tolist()

data = defaultdict(list) # with deafultdict function , our main work is done . 
#learn more about deafult dict at : https://www.educative.io/edpresso/learning-about-defaultdict-in-python

for i in iris:
  data[i[-1]].append(i[:-1]) # adding the list values to our default list, take the class as first , which is last in the list so we put-1 and  by cutting -1 in apending the features, beacuase , the last value is our class(0,1,2), we only need features in last . 
  # our classes(setosa,versicolor,virginca) are the last in the list , so thats y we are inputing -1 , so it will take -1 as a key(classs) and separate them . 
  
def knn(predict,data,k=3): # KNN Algo , a true real world example how maths help find distance b/w points and sort them . Simple as  that
  distance = []
  for group in data:
    for features in data[group]:
      eucd_dist = sqrt( (features[0]-predict[0])**2 + (features[1]-predict[1])**2 )
      distance.append([eucd_dist,group])
  
  votes = [i[1] for i in sorted(distance)[:5]]
  print(f"the votes are : {votes}")

  vote_result = Counter(votes).most_common()[0][0]
  return vote_result

ypred = 5.9,3,4.2,1.5 # = belongs to class 1(Versicolor) , we could have used Xtest , but you can use your own  way to test . try out taking features from the dataset .
knn(ypred,data,3)

 # Visualizing the data
plt.figure(figsize=(10,6))
[[plt.scatter(ii[0],ii[1],s=100,c="blue") for ii in data[i]] for i in data]
plt.scatter(ypred[0],ypred[1],color="g")
plt.show()

# It is a simple implementation of KNN Algorithm , i hope the basics created a base for you , you can add xtest and accuracy test to check the accyracy too. 
# Inspired from  SentDex | 


