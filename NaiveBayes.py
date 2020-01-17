
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:38:00 2019

@author: Rohan
"""

import csv
import sys


class NaiveBayes():
    
  #initialise
  def __init__(self):
      self.total = 0
      self.predictions = []
      
      
  #code to compute marginal probabilities when class=1 and class=0
  def marginal(self,y_train):
      pos,neg = 0,0      
      for i in range(len(y_train)):
          if(y_train[i] == '0'):
              neg+=1
          else:
              pos+=1

      self.total = pos+neg
      pos,neg = pos/self.total, neg/self.total
      return pos,neg
  
    
      
  #Train the NB model by computing the probabilities
  def training(self,f):
    values = []

    #read csv from file
    with open(f) as file:
      reader = csv.reader(file,delimiter = ',')
      count = 0
      for i in reader:
        if(count==0):
            attributes = [x for x in i if x!='class']
            count+=1
        else:
            values.append(i)
            count+=1

    #Convert raw data
    x_train = {z: [] for z in attributes}
    y_train = []
      
    #convert into dictionaries  
    for i in range(len(values)):
         for j in range(len(values[i])):
           if(j==len(values[i])-1):
              y_train.append(values[i][j])
           else:
              x_train[attributes[j]].append(values[i][j]) 
    
    #create two dictionaries - one when class is true and one when class is false
    self.cond_probs_given_false = {z: {} for z in x_train.keys()}
    self.cond_probs_given_true = {z: {} for z in x_train.keys()}


    #iterate over the featues to compute the conditional probabilities
    for i in x_train.keys(): 
    
         categories = set(x_train[i])
         for k in categories:
            self.cond_probs_given_false[i][k] = 0
            self.cond_probs_given_true[i][k] = 0
      
         for j in range(len(x_train[i])):
           for k in categories:
             if(x_train[i][j]==k and y_train[j] == '0'):
               self.cond_probs_given_false[i][k] += 1
             if(x_train[i][j]==k and y_train[j] == '1'):
               self.cond_probs_given_true[i][k] += 1
               
    
         total1,total2 = 0,0
         for k in categories:
             total1 += self.cond_probs_given_false[i][k]
             total2 += self.cond_probs_given_true[i][k]
      
         for k in categories:
             self.cond_probs_given_false[i][k] = self.cond_probs_given_false[i][k]/total1    
             self.cond_probs_given_true[i][k] = self.cond_probs_given_true[i][k]/total2

    
    return x_train,y_train  
   
    
            
  #Train the NB model by feeding test values  
  def test_model(self,f,pos,neg):
      
          #read test CSVs
          values2 = []
          with open(f) as file:
                reader = csv.reader(file,delimiter = ',')
                count = 0
                for i in reader:
                   if(count==0):
                      attributes = [x for x in i if x!='class']
                      count+=1
                   else:
                      values2.append(i)
                      count+=1

          x_test = {z: [] for z in attributes}
          y_test = []
          for i in range(len(values2)):
             for j in range(len(values2[i])):
                 if(j==len(values2[i])-1):
                      y_test.append(values2[i][j])
                 else:
                      x_test[attributes[j]].append(values2[i][j]) 


          test_dict ={}
          TP,FN,FP,TN = 0,0,0,0
          
          #iterate over each vector in the test file
          for i in range(len(y_test)):
              input_val = [x_test[attributes[x]][i] for x in range(len(attributes))]
              for j in range(len(attributes)):
                 test_dict[attributes[j]] = input_val[j]

    
              y_true,y_false = pos,neg
    
              for key in test_dict:
                 y_true = y_true*self.cond_probs_given_true[key][test_dict[key]]        
                 y_false = y_false*self.cond_probs_given_false[key][test_dict[key]] 
      

              T = (y_true/(y_true+y_false))
    
              if T>=0.5:
                 prediction = '1'
              else:
                 prediction = '0'

              #calculate confusion matrix
              if(y_test[i]==prediction and prediction == '1'):    
                 TP+=1
              elif(y_test[i]!=prediction and prediction == '1'):
                 FP+=1
              elif(y_test[i]==prediction and prediction == '0'):
                 TN+=1
              elif(y_test[i]!=prediction and prediction == '0'):
                 FN+=1

              self.predictions.append(prediction)
            
          
          print("\n")
          print("Accuracy:  ",(TP+TN)/(TP+FP+FN+TN))
          print("\n")
          
          return TP,FN,FP,TN
  


if __name__== "__main__":   
   obj = NaiveBayes()
   x_train,y_train = obj.training(sys.argv[1])

   pos,neg = obj.marginal(y_train)

   TP,FN,FP,TN = obj.test_model(sys.argv[2],pos,neg)
   
   print("The Confusion Matrix: ")
   print("\n")
   print("                                 Predicted         ")
   print("                     |    Class=1           Class=0 ")
   print("           Class=1   |     {0}                 {1} ".format(TP,FN))
   print(" Actual    Class=0   |     {0}                 {1} ".format(FP,TN))
   print("\n\n\n")
