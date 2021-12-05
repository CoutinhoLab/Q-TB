import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
from molvs import standardize_smiles
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
import numpy as np


    


# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model building
def build_model(input_data):
    # Reads in saved Classification model   
    load_model = pickle.load(open('app.pkl', 'rb'))
    # Apply model to make predictions
    prediction = load_model.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(prediction, name='Bioactivity Class')    
    df = pd.concat([ prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)

# Logo image
image = Image.open("App.jpg")

st.image(image, use_column_width=True)

# Page title
st.markdown("""
# Class Prediction App
This WebApp allows to predict the Bioactivity Class of a molecule, as QcrB inhibitors of Mycobacterium tuberculosis.
- Upload the input file using the name "Test.csv"
- **Credits**:  App built in `Python` and  `Streamlit` by Afreen Khan.
- Read the full Paper(https://doi.org/..).
---
""")

# Molecular descriptor calculator
class ECFP6:
    def __init__(self, smiles):
        self.mols = [Chem.MolFromSmiles(i) for i in smiles]
        self.smiles = smiles

    def mol2fp(self, mol, radius = 3):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = radius)
        array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, array)
        return array

    def compute_ECFP6(self, name):
        bit_headers = ['bit' + str(i) for i in range(2048)]
        arr = np.empty((0,2048), int).astype(int)
        for i in self.mols:
            fp = self.mol2fp(i)
            arr = np.vstack((arr, fp))
        df_ecfp6 = pd.DataFrame(np.asarray(arr).astype(int),columns=bit_headers)
        #df_ecfp6.insert(loc=0, column='Smiles', value=self.smiles)
        df_ecfp6.to_csv('Desc_out.csv', index=False)
        
        
def main1():
    
    filename = 'Test.csv'
    df = pd.read_csv(filename,sep=',')
    smiles = [standardize_smiles(i) for i in df['Smiles'].values]

    ecfps_descriptor = ECFP6(smiles)
    ecfps_descriptor.compute_ECFP6(filename)
if __name__ == '__main__':
    main1()


# Sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['csv'])
    st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/..)
""")

if st.sidebar.button('Predict'):
    load_data = pd.read_csv(uploaded_file, sep=',')

    st.header('**Original input data**')
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):
        main1()

    # Read in calculated descriptors and display the dataframe
    st.header('**Calculated molecular descriptors**')
    desc = pd.read_csv('Desc_out.csv')
    st.write(desc)
    st.write(desc.shape)

    # Apply trained model to make prediction on query compounds
    build_model(desc)

else:
    st.info('Upload input data in the sidebar to start!')