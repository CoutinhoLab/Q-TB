# Q-TB
Refer the Manual for detailed steps.

1)	Create a conda environment 
a) At first, we’ll create a conda called QTB
conda create -n QTB python=3.7.9 
b)	Now log in to the QTB environment 
conda activate QTB 
2)	Install prerequisite libraries
a)	Download requirements.txt file
wget https://raw.githubusercontent.com/...../requirements.txt 
b)	Pip install libraries
pip install -r requirements.txt
3)	Download and unzip contents from GitHub repo
Download and unzip contents from https://github.com/....
4)	Generating the Pickle (PKL) file
The machine learning model used in this web app will firstly have to be generated by successfully running the included Jupyter notebook bioactivity_prediction_app.ipynb. Upon successfully running all code cells, a pickled model called App.pkl will be generated.
5)	Launch the app
streamlit run app.py
