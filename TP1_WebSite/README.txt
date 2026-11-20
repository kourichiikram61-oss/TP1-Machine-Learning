TP1 WEB APPLICATION
===================

1) Install Python 3.10 or newer.
2) Extract this folder.
3) Double-click start.bat.
   The script installs the required packages, trains/saves the models, then opens the website.

Or run manually in Anaconda Prompt / terminal:
   cd path\to\TP1_WebSite
   python -m pip install -r requirements.txt
   python train_models.py
   python -m streamlit run app.py

The app provides:
- Diabetes Simple Linear Regression (BMI -> progression score)
- Diabetes Multiple Linear Regression (10 features -> progression score)
- Iris Simple Linear Regression (petal length -> sepal length)
- Iris Multiple Linear Regression (3 features -> sepal length)
- Iris Classification (4 measurements -> Setosa / Versicolor / Virginica), with a reference flower image

IMPORTANT:
- The Diabetes dataset supplied by Scikit-learn is already normalized. Inputs must use the normalized scale shown in the interface; these are not ordinary clinical values.
- The original Iris regression models predict a numeric sepal length, not a flower species. The separate Iris Classification model was added to meet the requirement of identifying the flower and displaying its representative image.
- Flower images are loaded from Wikimedia Commons, so internet access is needed to display them.
- The diabetes output is an experimental dataset prediction, not a medical diagnosis.
