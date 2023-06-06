"""
@author: 
"""
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

import pickle
import streamlit as st

# loading the saved models
model = pickle.load(open('big_mart_model.pkl', 'rb'))

# page title
st.title("Gbemi's Sales Prediction using ML")

# #Image
st.image('3D code.png')
# st.divider() 
# # getting the input data from the user
col1, col2, col3 = st.columns(3)

with col1:
    Item_Visibility = st.number_input('**:blue[Item _Visibility_]**', min_value=0.00, max_value=0.40, step=0.01)

with col1:
    Item_MRP = st.number_input('**:blue[Item _MRP_]**', min_value=30.00, max_value=270.00, step=1.00)

with col2:
    Item_Fat_Content = st.selectbox('**:blue[Item _Fat Content_]**', ['Low Fat', 'Regular'])

with col2:
    Outlet_Size = st.selectbox('**:blue[Outlet _Size_]**', ['Small', 'Medium', 'High'])

with col3:
    Outlet_Location_Type = st.selectbox('**:blue[Outlet _Location Type_]**', ['Tier 1', 'Tier 2', 'Tier 3'])

# #Data Preprocessing
    
data = {
        'Item_Visibility': Item_Visibility,
        'Item_MRP' : Item_MRP,
        'Outlet_Size' : Outlet_Size,
        'Outlet_Location_Type_Numbers' : Outlet_Location_Type,
        'Item_Fat_Content_Regular': Item_Fat_Content
            }

oe = OrdinalEncoder(categories = [['Small','Medium','High']])
scaler = StandardScaler()

def make_prediction(data):
    df = pd.DataFrame(data, index=[0])
    
    if df['Item_Fat_Content_Regular'].values == 'Low Fat':
        df['Item_Fat_Content_Regular'] = 0.0

    if df['Item_Fat_Content_Regular'].values == 'Regular':
        df['Item_Fat_Content_Regular'] = 1.0

    df['Outlet_Location_Type_Numbers'] = df['Outlet_Location_Type_Numbers'].str.extract('(\d+)', expand=False)
    df['Outlet_Size'] = oe.fit_transform(df[['Outlet_Size']])
    df[['Item_Visibility', 'Item_MRP']] = StandardScaler().fit_transform(df[['Item_Visibility', 'Item_MRP']])
    
    prediction = model.predict(df)
    
    return round(float(prediction),2)
    

# # creating a button for Prediction
sales_prediction_output = ""


if st.button('**Predict Sales**'):
    sales_prediction = make_prediction(data)
    sales_prediction_output = f"**:blue[The sales is predicted to be {sales_prediction}]**"
    with st.spinner('Predicting Sales...'):
        time.sleep(1)
        st.success(sales_prediction_output, icon="📊")
# st.divider() 