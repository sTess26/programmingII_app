import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import streamlit as st

s = pd.read_csv("social_media_usage.csv")
def clean_sms(x):
    x = np.where(x == 1,1,0)
    return(x)
ss = pd.DataFrame({

    #sm_li (binary variable which indicates whether or not the individual uses LinkedIn --> 0 = don't use, 1 = use)
    "sm_li":clean_sms(s["web1h"]).astype(int),

    #income (ordered numeric from 1 to 9, above 9 considered missing)
    "income": np.where(s["income"] > 9, np.nan, s["income"]),

    #education (ordered numeric from 1 to 8, above 8 considered missing)
    "educ":np.where(s["educ2"] > 8, np.nan, s["educ2"]),

    #parent (binary) --> 1 Yes, 1 No, else NA
    "par":np.where(s["par"] > 2, np.nan, 
                           np.where(s["par"]==1,1,0)),

    #married (binary) --> 1 Married, 0 Not married, else NA
    "marital":np.where(s["marital"] > 7, np.nan, 
                       np.where(s["marital"]==1,1,0)),

    #female (binary) --> 1 Female, 0 Not female (Male or Other), else NA
    "gender":np.where(s["gender"] > 4, np.nan, 
                      np.where(s["gender"]==2,1,0)),
    
    #age (numeric, above 98 considered missing)
    "age":np.where(s["age"] > 98, np.nan, s["age"])}).dropna(ignore_index=True)

ss["sm_li"] = ss["sm_li"].astype(int)
ss["income"] = ss["income"].astype(int)
ss["educ"] = ss["educ"].astype(int)
ss["age"] = ss["age"].astype(int)
ss["gender"] = ss["gender"].astype(int).astype("category")
ss["par"] = ss["par"].astype(int).astype("category")
ss["marital"] = ss["marital"].astype(int).astype("category")

y = ss["sm_li"]
x = ss[["income", "educ", "age", "gender","par","marital"]]

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    stratify=y,      
                                                    test_size=0.2,   
                                                    random_state=987)

# Initialize algorithm
lr = LogisticRegression(class_weight='balanced', random_state=987)

# Fit algorithm to training data
lr.fit(x_train, y_train)

#function to create peson based on inputs, making sure binary vars are categorical
def create_person(income, educ, age, gender, par, marital):
    # Create a pandas DataFrame or Series with the correct categories
    person = pd.DataFrame({
        'income': [income],
        'educ': [educ],
        'age': [age],
        'gender': pd.Categorical([gender], categories=[0, 1]),
        'par': pd.Categorical([par], categories=[0, 1]),
        'marital': pd.Categorical([marital], categories=[0, 1])
    })
    return person

def linkedin_class(income, educ, age, gender, par, marital):

    # Load packages
    from sklearn.linear_model import LogisticRegression
    import plotly.graph_objects as go # data visualization package


    #create person
    person = create_person(income, educ, age, gender, par, marital)

    # Make predictions using the logistic regression model
    predicted_class = lr.predict(person)[0]
    probs = lr.predict_proba(person)[0][1]

    if predicted_class == 0:
        return("Not a LinkedIn User")
    else: 
        return("LinkedIn User")
    return(predicted_class)

def linkedin_app(income, educ, age, gender, par, marital):

    # Load packages
    from sklearn.linear_model import LogisticRegression
    import plotly.graph_objects as go # data visualization package


    #create person
    person = create_person(income, educ, age, gender, par, marital)

    # Make predictions using the logistic regression model
    predicted_class = lr.predict(person)[0]
    probs = lr.predict_proba(person)[0][1]

    #labels
    if probs > 0.75:
        label = "Very High"
    elif probs > 0.5:
        label = "Moderate - High"
    elif probs > 0.25:
        label = "Moderate - Low"
    else:
        label = "Very Low"

    if predicted_class == 1:
        class_label = "LinkedIn User"
    else:
        class_label = "Not a LinkedIn User"

    # Plot likelihood on gauge plot
    fig = go.Figure(go.Indicator(
       mode = "gauge+number",
       value = probs *100,
       title = {'text': f"<b>Probability of Using LinkedIn: {label}</b> <br> <i>Probabilities as Percentages</i>"},
       gauge = {"axis": {"range": [0, 100],
                "tickvals": [0, 25, 50, 75, 100]},
                    "steps": [
                        {"range": [0, 25], "color":"white"},
                        {"range": [25, 50], "color":"lightblue"},
                        {"range": [50, 75], "color":"royalblue"},
                        {"range": [75, 100], "color":"darkblue"}],
                "bar":{"color":"indianred"}}))
    return st.plotly_chart(fig)


###UI for streamlit app

st.markdown("##### Enter a person's characteristics to see their likelihood of using LinkedIn!")

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    Income = st.selectbox("Select an Income Level:", ["Less than $10,000", "$10,000 - $20,000", "$20,000 - $30,000", "$30,000 - $40,000", "$40,000 - $50,000", "$50,000 - $75,000", "$75,000 - $100,000", "$100,000 - $150,000", "Above $150,000"])
with col2:
    Education = st.selectbox("Select an Education Level:", ["Less than highschool", "Highschool incomplete", "Highschool graduate", "Some college, no degree", "Two-year associate degree", "Four-year college or university degree"])
with col3:
    Age = st.slider("Select an Age:", 18,97)

col4, col5, col6 = st.columns(3)
with col4:
    Gender = st.selectbox("Select a Gender:", ["Female","Not Female"])
with col5:
    Parent = st.selectbox("Select a Parental Status:", ["Parent","Not a Parent"])
with col6:
    Married = st.selectbox("Select a Marital Status:", ["Married","Not Married"])


if Income == "Less than $10,000":
    Income = 1
elif Income =="$10,000 - $20,000":
    Income = 2
elif Income =="$20,000 - $30,000":
    Income = 3
elif Income =="$30,000 - $40,000":
    Income = 4
elif Income =="$40,000 - $50,000":
    Income = 5
elif Income =="$50,000 - $75,000":
    Income = 6
elif Income =="$75,000 - $100,000":
    Income = 7
elif Income =="$100,000 - $150,000":
    Income = 8
elif Income =="Above $150,000":
    Income = 9

if Education == "Less than highschool":
    Education = 1
elif Education =="Highschool incomplete":
    Education = 2
elif Education =="Highschool graduate":
    Education = 3
elif Education =="Some college, no degree":
    Education = 4
elif Education =="Two-year associate degree":
    Education = 5
elif Education =="Four-year college or university degree":
    Education = 6

if Gender == "Female":
    Gender = 1
elif Gender =="Not Female":
    Gender = 0

if Married == "Married":
    Married = 1
elif Married =="Not Married":
    Married = 0

if Parent == "Parent":
    Parent = 1
elif Parent =="Not a Parent":
    Parent = 0


st.markdown("---")
st.markdown(f"<p style='color:indianred; font-weight:bold;'>This person's predicted class is: {linkedin_class(Income, Education, Age, Gender, Parent, Married)}</p>", unsafe_allow_html=True)
linkedin_app(Income, Education, Age, Gender, Parent, Married)

st.markdown("---")
