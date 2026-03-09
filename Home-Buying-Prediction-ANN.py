#import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model


#create a func for cleaning
def preprocess_data(path):
    
    df=pd.read_csv(path)
    
    #create coloumn
    df["Buy_Home"]=((df['Disposable_Income']>=20000)&(df['Desired_Savings']>=10000)).astype(int)
    
    #convert numerical
    
    df=pd.get_dummies(df,columns=['Occupation','City_Tier'])
    
    X=df.drop("Buy_Home",axis=1)
    y=df["Buy_Home"]
    
    #scaling
    
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    
    joblib.dump(scaler,"...../models/scaler.pkl")
    joblib.dump(X.columns,"..../models/feature_columns.pkl")
    
    return X_scaled,y

#assign X,y
X,y=preprocess_data("...\salary and expense.csv")

# split X,y
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)

#crate model and fit

model=Sequential()
model.add(Dense(32,activation='relu',input_dim=X_train.shape[1]))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=50,batch_size=32)

#calculate loss and acc
loss,acc=model.evaluate(X_test,y_test)

print("model Accuracy :",acc)


model.save("..../models/Buy_model.h5")

model=load_model("....\models\Buy_model.h5")
scaler=joblib.load("..../models/scaler.pkl")
feature_columns = joblib.load(".../models/feature_columns.pkl")



income                     =int(input("Enter income :"))
age                        =int(input("Enter Age:"))
dependents                 =int(input("Dependents (0-4):"))
occupation                 =input("Select Occupation (Self_Employed / Retired /Student /Professional):")
city_tier                  =input("Select City_Tier (Tier_1 / Tier_2 /Tier_3) :")
rent                       =int(input("Enter Rent:"))
loan                       =int(input("Enter Loan_Repayment:"))
insurance                  =int(input("Insurance:"))
groceries                  =int(input("Enter Groceries:"))
transport                  =int(input("Enter Transport:"))
eating_out                 =int(input("Enter Eating_Out:"))
entertainment              =int(input("Enter Entertainment:"))
utilities                  =int(input("Enter Utilities:"))
healthcare                 =int(input("Enter Healthcare:"))
education                  =int(input("Enter Education:"))
misc                       =int(input("Enter Miscellaneous:"))
saving_percent            =int(input("Enter Desired_Savings_Percentage:"))
desired_savings            =int(input("Enter Desired_Savings:"))
disposable_income          =int(input("Enter Disposable_Income:"))




sample = pd.DataFrame([{
"Income":income,
"Age":age,
"Dependents":dependents,
"Occupation":occupation,
"City_Tier":city_tier,
"Rent":rent,
"Loan_Repayment":loan,
"Insurance":insurance,
"Groceries":groceries,
"Transport":transport,
"Eating_Out":eating_out,
"Entertainment":entertainment,
"Utilities":utilities,
"Healthcare":healthcare,
"Education":education,
"Miscellaneous":misc,
"Desired_Savings_Percentage":saving_percent,
"Desired_Savings":desired_savings,
"Disposable_Income":disposable_income,
}])

sample = pd.get_dummies(sample)
sample = sample.reindex(columns=feature_columns, fill_value=0)

sample_scaled=scaler.transform(sample)

prediction=model.predict(sample_scaled)
    

print("Probability of buying home:", round(prediction[0][0],3))
    
if prediction[0][0] > 0.5:
    print("He can afford to buy home")
else:
    print("He cannot buy home")



    

