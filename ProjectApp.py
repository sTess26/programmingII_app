import pandas as pd
import numpy as np
import altair as alt
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

ss["sm_li"] = ss["sm_li"].astype(int).astype("category")
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

def linkedin_app(income, educ, age, gender, par, marital):

    #warning messages if numbers are out of range
    if income > 9 or income == 0 or educ > 6 or educ == 0 or age < 18 or age > 97 or gender > 1 or par > 1 or marital > 1:
        st.write("## WARNING")
    if income > 9 or income == 0:
        st.write("Please enter a valid income (integer between 1 and 9)")
    if educ > 6 or educ == 0:
        st.write("Please enter a valid education level (integer between 1 and 6)")
    if age < 18 or age > 97:
        st.write("Please enter a valid age (integer between 18 and 97)") #using 18 because that is the lowest age in df
    if gender > 1:
        st.write("Please enter a valid gender (0 or 1)")
    if par > 1:
        st.write("Please enter a valid parental status (0 or 1)")
    if marital > 1:
        st.write("Please enter a valid marital status (0 or 1)")   

    #fig will not print if any vars are out of range
    if income > 9 or income == 0 or educ > 6 or educ == 0 or age < 18 or age > 97 or gender > 1 or par > 1 or marital > 1:
        return

 
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
       value = probs,
       title = {'text': f"Probability of Using LinkedIn: {label}<br>Predicted Class: {class_label}"},
       gauge = {"axis": {"range": [0, 1]},
                    "steps": [
                        {"range": [0, 0.25], "color":"red"},
                        {"range": [0.25, 0.5], "color":"lightcoral"},
                        {"range": [0.5, 0.75], "color":"lightgreen"},
                        {"range": [0.75, 1], "color":"green"}],
                "bar":{"color":"grey"}}))
    return st.plotly_chart(fig)


st.markdown("##### Enter a person's characteristics to see their likelihood of using LinkedIn!")
st.markdown("###### Definitions for each variable are at the bottom of the page")

Income = st.number_input("Enter Income:", value=5)
Education = st.number_input("Enter Education Level:", value=5)
Age = st.number_input("Enter Age:", value=30)
Gender = st.number_input("Enter Gender:", value=1)
Parent = st.number_input("Enter Parental Status:", value=0)
Married = st.number_input("Enter Marital Status:", value=0)

linkedin_app(Income, Education, Age, Gender, Parent, Married)

st.markdown("### Variable Key")
st.markdown("##### Income")
st.markdown("""
- 1: Less than 10,000
- 2: 10,000 - 20,000
- 3: 20,000 - 30,000
- 4: 30,000 - 40,000
- 5: 40,000 - 50,000
- 6: 50,000 - 75,000
- 7: 75,000 - 100,000
- 8: 100,000 - 150,000
- 9: 150,000 and above
""")
st.markdown("##### Education")
st.markdown("""
- 1: Less than highschool
- 2: Highschool incomplete
- 3: Highschool graduate
- 4: Some college, no degree
- 5: Two-year associate degree from college or university
- 6: Four-year college or university degree
""")
st.markdown("##### Age")
st.markdown("""
- 18 through 96: Numeric Age
- 97: 97 years or above
""")
st.markdown("##### Gender")
st.markdown("""
- 0: Not Female (Male or Other)
- 1: Female
""")
st.markdown("##### Parental Status")
st.markdown("""
- 0: Not a Parent
- 1: Parent
""")
st.markdown("##### Marital Status")
st.markdown("""
- 0: Not Married (Living with partner, divorced, separated, widowed, never married)
- 1: Married
""")