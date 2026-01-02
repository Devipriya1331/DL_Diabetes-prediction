'''
1.Number of times pregnant
2.Plasma glucose concentration a 2 hrs in an oral glucose tolerance test
3.Diagnostic blood pressure(mm Hg)
4.Tricepts skin fold thickness(mm)
5.2-hour serum insulin (mm U/ml)
6.Body mass index (weight in kg/(height in m)^2)
7.Diabetes pedigree function
8.Age(years)
10.Class variable (0 or 1)
 
'''

from numpy import loadtxt
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense

dataset=loadtxt('pima-indians-diabetes.csv',delimiter=',')
print(dataset)

x=dataset[: , 0:8]
y=dataset[: , 8]
print('Input',x)
print('Output',y)

model=Sequential()

#adding layers

model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#compilation

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#model training

model.fit(x,y,epochs=10,batch_size=10)

#evaluation

_,accuracy=model.evaluate(x,y)
print('Accuracy: %.2f' % (accuracy*100))

##model save

model_json=model.to_json()
with open("model.json","w")as json_file:
    json_file.write(model_json)
model.save_weights("model.weights.h5")
print("Saved model to disk")

