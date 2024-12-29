import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle

def create_model(data):
    X = data.drop(['diagnosis'], axis = 1)
    Y = data['diagnosis']

    #scaling the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
   # Y = scaler.fit_transform(Y)  we dont need to fit y because it is the labels it has only 0 for B and 1 for M

    #splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #training the model
    model = LogisticRegression()
    model.fit(X_train, Y_train)

     #test the model
    
    X_predict = model.predict(X_test)
    print('Accuracy of the model is:',accuracy_score(Y_test, X_predict))
    print("Classification report:\n",classification_report(Y_test, X_predict))
    

    return model, scaler

    



def get_clean_data():
    data = pd.read_csv("data/data.csv")
    #print(data.head())

    data = data.drop(['Unnamed: 32','id'], axis=1)

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B':0})
    
    

    return data

def main():
    data = get_clean_data()
   
    #print(data.info()) 

    model,scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f:
     pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
       pickle.dump(scaler, f)

if __name__ == '__main__':
    main()