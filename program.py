

from google.colab import drive
drive.mount('/content/gdrive')
import pandas as pd
df=pd.read_csv('gdrive/My Drive/CM1.csv')
kkk=0
for i in df['Defective']:
  if i=="Y":
     nnn=1.0
  elif i=="N":
     nnn=0.0
  df['Defective'][kkk]=nnn
  kkk+=1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.utils import to_categorical

import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD,Adam
!pip install deap
from deap import creator,tools,base
np.random.seed(seed=9)

import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

IND_INIT_SIZE = 36
MAX_ITEM = 36
MAX_WEIGHT = 50
NBR_ITEMS = 2

# To assure reproductibility, the RNG seed is set prior to the items
# dict initialization. It is also seeded in main().
random.seed(64)


creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.Fitness)
def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    :
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2
toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_item", random.randrange, NBR_ITEMS)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_item, IND_INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
def fun( individual):
    
    np.random.seed(seed=90)
    dataset= df.iloc[:,1:37].values
    y = df.iloc[:,37].values
    indexes=df.columns
    X= pd.DataFrame(dataset)
    print(X.shape)
   
    encoder =  LabelEncoder()
    y1 = encoder.fit_transform(y)
    Y = pd.get_dummies(y1).values
    cols = [index for index in range(len(individual)) if individual[index] == 0]
    
    X1 = X.drop(cols, axis=1)
    print(X1.shape)
    noOfCol=X1.shape
    noOfCols=noOfCol[1]
    print(noOfCols)
    X2 = pd.get_dummies(X1)
    X_train,X_test, y_train,y_test = train_test_split(X2,Y,test_size=0.25,random_state=30)
    X_train = preprocessing.scale(X_train)
    X_trainOhFeatures = X_train
    X_testOhFeatures = X_test
    X_trainOhFeatures= pd.DataFrame(X_trainOhFeatures)
    X_testOhFeatures=pd.DataFrame(X_testOhFeatures)

#     # Remove any columns that aren't in both the training and test sets
#     sharedFeatures = set(X_trainOhFeatures.columns) & set(X_testOhFeatures.columns)
#     removeFromTrain = set(X_trainOhFeatures.columns) - sharedFeatures
#     removeFromTest = set(X_testOhFeatures.columns) - sharedFeatures
#     X_trainOhFeatures = X_trainOhFeatures.drop(list(removeFromTrain), axis=1)
#     X_testOhFeatures = X_testOhFeatures.drop(list(removeFromTest), axis=1)
    model = Sequential()
    model.add(Dense(258, input_shape=(noOfCols,),
                ))
#     model.add(Dropout(0.5))
    model.add(Dense(50,activation='softmax'))
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='relu'))
    # Compile mod
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    history = model.fit(X_train, y_train, validation_split=0.3, epochs=20, verbose=2)
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#     # summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
    return scores[0],scores[1]

toolbox.register("evaluate", fun)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

def main():
    random.seed(64)
    NGEN = 5
    MU = 10
    LAMBDA = 10
    CXPB = 0.5
    MUTPB = 0.37
    
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(2, similar=numpy.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              )
    
    return pop, stats,hof,toolbox
                 
if __name__ == "__main__":
    a=main()

?print(toolbox.selBest(pop))

print(a)

pop=a[0]

print(pop)

print(tools.selBest(pop,1))

def ind2plot( individual):
    
    np.random.seed(seed=90)
    dataset= df.iloc[:,1:37].values
    y = df.iloc[:,37].values
    indexes=df.columns
    X= pd.DataFrame(dataset)
    print(X.shape)
   
    encoder =  LabelEncoder()
    y1 = encoder.fit_transform(y)
    Y = pd.get_dummies(y1).values
    cols = [index for index in range(len(individual)) if individual[index] == 0]
    
    X1 = X.drop(cols, axis=1)
    print(X1.shape)
    noOfCol=X1.shape
    noOfCols=noOfCol[1]
    print(noOfCols)
    X2 = pd.get_dummies(X1)
    X_train,X_test, y_train,y_test = train_test_split(X2,Y,test_size=0.25,random_state=30)
    X_train = preprocessing.scale(X_train)
    X_trainOhFeatures = X_train
    X_testOhFeatures = X_test
    X_trainOhFeatures= pd.DataFrame(X_trainOhFeatures)
    X_testOhFeatures=pd.DataFrame(X_testOhFeatures)

#     # Remove any columns that aren't in both the training and test sets
#     sharedFeatures = set(X_trainOhFeatures.columns) & set(X_testOhFeatures.columns)
#     removeFromTrain = set(X_trainOhFeatures.columns) - sharedFeatures
#     removeFromTest = set(X_testOhFeatures.columns) - sharedFeatures
#     X_trainOhFeatures = X_trainOhFeatures.drop(list(removeFromTrain), axis=1)
#     X_testOhFeatures = X_testOhFeatures.drop(list(removeFromTest), axis=1)
    model = Sequential()
    model.add(Dense(258, input_shape=(noOfCols,),
                ))
#     model.add(Dropout(0.5))
    model.add(Dense(50,activation='softmax'))
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='relu'))
    # Compile mod
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    history = model.fit(X_train, y_train, validation_split=0.3, epochs=20, verbose=2)
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return scores[0],scores[1]

bestChromosome=tools.selBest(pop,1)

ind2plot(bestChromosome)

