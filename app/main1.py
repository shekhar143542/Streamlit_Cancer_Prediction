from sklearn.metrics import confusion_matrix
import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import csv
import streamlit.components.v1 as components


    
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    #print(data.head())

    data = data.drop(['Unnamed: 32','id'], axis=1)

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B':0})
    
    
    return data 

def switch_theme():
    # Sidebar option for choosing theme
    theme_choice = st.sidebar.radio("Choose a Theme", ["Dark", "Light"])

    if theme_choice == "Dark":
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #181818;
                color: white;
            }
            .sidebar .sidebar-content {
                background-color: #181818;
                color: white;
            }
            .stTextInput>div>div>input, .stSelectbox>div>div>div {
                background-color: #333333;
                color: white;
            }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: white;
                color: black;
            }
            .sidebar .sidebar-content {
                background-color: white;
                color: black;
            }
            .stTextInput>div>div>input, .stSelectbox>div>div>div {
                background-color: #FFFFFF;
                color: black;
            }
            </style>
            """, unsafe_allow_html=True)



def add_sidebar():

    st.sidebar.header("Cell Nuclei Mearsurements")
    data = get_clean_data()
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label,key in slider_labels: 
        input_dict[key] = st.sidebar.slider(

            label,
            min_value = float(0),
            max_value = float(data[key].max()),
            value = float(data[key].mean()),
            step = 0.01,
            help="Adjust this slider to change the measurement value for " + label
        )
    return input_dict

def get_scaled_data(input_dict):
    data = get_clean_data()

    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_value = X[key].max()
        min_value = X[key].min()
        scaled_value = (value-min_value)/(max_value-min_value)
        scaled_dict[key] = scaled_value
    
    return scaled_dict
#we can also use this normalization from sklearn it is working but graph is not showing any readings we need to modify it 
   # normal = MinMaxScaler()
    #scaled_data = normal.fit_transform(X)
    #scaled_df = pd.DataFrame(scaled_data, columns=X.columns)

   # return scaled_df
   

def get_radar_chart(input_data):

    input_data = get_scaled_data(input_data)

    categories = ['Radius','Texture','Perimeter','Area','Smoothness','Compactness','Concavity','Concave Points','Symmetry','Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
      r=[
          
        input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
                input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
                input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
                input_data['fractal_dimension_mean']
                
                ],
      
      theta=categories,
      fill='toself',
      name='Mean Value'
))
    fig.add_trace(go.Scatterpolar(
      r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
                
                ],
      theta=categories,
      fill='toself',
      name='Standard Error'
))
    
    fig.add_trace(go.Scatterpolar(
      r=[
           input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
                input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
                input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
                input_data['fractal_dimension_worst']
         ],
      theta=categories,
      fill='toself',
      name='Worst'
))

    fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
  showlegend=True
)

    st.plotly_chart(fig)

def plot_prediction_probability(prob_benign, prob_malacious):
    # Create the gauge chart for benign probability
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_benign,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probability of Benign"},
        gauge={'axis': {'range': [None, 1]}}
    ))

    fig1.update_layout(height=300, width=400)
   

    # Create the gauge chart for malignant probability
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_malacious,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probability of Malacious"},
        gauge={'axis': {'range': [None, 1]}}
    ))

    col1, col2 = st.columns(2)

    
    fig2.update_layout(height=300, width=400)
    
    with col1:
        st.plotly_chart(fig1)

    with col2:
        st.plotly_chart(fig2)
    



def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is:")
    #st.write(input_array_scaled)
    if prediction[0] == 0:
        st.write("<span class = 'diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class = 'diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
    
    #st.write(prediction) just to test working correctly or not
    proba_benign = model.predict_proba(input_array_scaled)[0][0]
    proba_malacious = model.predict_proba(input_array_scaled)[0][1]

    st.write("<span class = 'probability1'>probability of being benign:</span>",proba_benign ,unsafe_allow_html=True)
    
    st.write("<span class = 'probability2'>probability of being malacious:</span>",proba_malacious,unsafe_allow_html=True)

    st.write("<span class = 'text'>This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.</span>",unsafe_allow_html=True)

    

    download_results(prediction,proba_benign,proba_malacious)

    return proba_benign, proba_malacious


def download_results(prediction,proba_benign, proba_malacious ):
    results = {
        'prediction':prediction,
        'probability of benign':proba_benign,
        'probability of malaciious':proba_malacious

    }
    with open('prediction_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Prediction', 'Probability of Benign', 'Probability of Malignant'])
        writer.writerow([prediction, proba_benign, proba_malacious])
    
    st.download_button(
        label="Download Results",
        data=open('prediction_results.csv', 'rb').read(),
        file_name="prediction_results.csv",
        mime="text/csv"
    )

def assess_risk(age, gender):
    # Example risk assessment rules (these can be adjusted based on more complex logic)
    if gender == 'Female':
        if age < 40:
            return "Your risk of breast cancer is low."
        elif 40 <= age <= 60:
            return "Your risk of breast cancer is moderate. Regular screening is recommended."
        else:
            return "Your risk of breast cancer is high. Consider regular screening and a healthcare consultation."
    elif gender == 'Male':
        if age < 50:
            return "Your risk of prostate cancer is low."
        elif 50 <= age <= 70:
            return "Your risk of prostate cancer is moderate. Regular screening is recommended."
        else:
            return "Your risk of prostate cancer is high. Consider regular screening and a healthcare consultation."
    else:
        return "Gender not recognized for risk assessment."
   

def main():
    st.set_page_config(
        page_title= "Breast Cancer Predictor",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    switch_theme()

    with open("app/assets/styles.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html= True)
        
    #input_data = add_sidebar()
    
    input_data = add_sidebar()
    #if st.button("Go to ChatBot"):
        #chatbot()
        #return
    
    st.sidebar.title("Help & Tutorials")

    # Button to show Instructions
    

    #st.write(input_data)
    with st.container():
        
        st.title("Breast Cancer Predictor")
        st.write("please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignanat(malacious) bases on the measurements it recieves from your cytosis lab. You can also update the measurements by hand using the sliders in the slidebar")

    col1, col2 = st.columns([4,1])
    #col1.button("Button in the larger column")
    #col2.button("Button in the smaller column")

    with col2:
         proba_benign, proba_malacious = add_predictions(input_data)
         if st.sidebar.button("How to Use the App"):
          st.sidebar.markdown("""
        ## How to Use the App
        
        This app allows you to:
        1. **Assess your cancer risk** based on your **age** and **gender**.
        2. **Predict whether a tumor is Malignant or Benign** by uploading a **CSV file** containing tumor-related data.
        
        ### Steps to use the app:
        - **Enter your age and gender** to see your cancer risk.
        - **Upload your data** in CSV format to get a prediction of whether a tumor is **Malignant** or **Benign**.
        """)

    # Button to show Malignant vs Benign Explanation
    if st.sidebar.button("What is Malignant and Benign?"):
        st.sidebar.markdown("""
        ## Malignant vs Benign Tumors
        
        - **Malignant Tumors**: Cancerous tumors that can spread to other parts of the body. They are harmful and need immediate treatment.
        - **Benign Tumors**: Non-cancerous tumors that do not spread. While generally harmless, they might still need to be monitored or removed.
        
        The app helps to predict if a tumor is **Malignant** or **Benign** based on tumor features from the uploaded data.
        """)

    # Button to show Risk Assessment Details
    if st.sidebar.button("Risk Assessment (Age & Gender)"):
        st.sidebar.markdown("""
        ## Risk Assessment (Age & Gender)
        
        This feature calculates your cancer risk based on your **age** and **gender**:
        - **For Women**: The risk is assessed for **Breast Cancer**.
        - **For Men**: The risk is assessed for **Prostate Cancer**.
        
        Enter your **age** and **gender** below to see your personalized cancer risk.
        """)

    with col1:
       # st.write("column1")
       get_radar_chart(input_data)
       plot_prediction_probability(proba_benign, proba_malacious)
       

    #get_bar_chart(input_data)
       st.title("Cancer Risk Prediction & Assessment")
    st.markdown("""
    ## Disclaimer

    **Important:** This app is intended for educational purposes only. While it uses machine learning to predict whether a tumor is **Malignant** or **Benign**, the predictions are based on historical data and may not always be accurate. **This app is not a substitute for professional medical advice, diagnosis, or treatment**. Always consult with a qualified healthcare professional for any health concerns or to confirm any diagnosis.
    
    The predictions made by this app are **not 100% reliable** and should not be used for making real-life medical decisions.
""")
    # Age and Gender Inputs
    age = st.number_input("Enter your age:", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Select your gender:", ['Male', 'Female'])
    
    # Display Risk Assessment
    risk_message = assess_risk(age, gender)
    st.subheader(f"Risk Assessment for Age: {age} and Gender: {gender}")
    st.write(risk_message)
    st.markdown("---")

    # BMI Calculator Section
    st.header("BMI Calculator")
    st.write("Calculate your Body Mass Index (BMI) below:")

    weight = st.number_input("Enter your weight (kg):", min_value=1.0)
    height = st.number_input("Enter your height (m):", min_value=0.1)
    if st.button("Calculate BMI"):
        bmi = weight / (height ** 2)
        st.success(f"Your BMI is {bmi:.2f}")
        if bmi < 18.5:
            st.info("Underweight")
        elif 18.5 <= bmi < 24.9:
            st.info("Normal weight")
        elif 25 <= bmi < 29.9:
            st.info("Overweight")
        else:
            st.warning("Obesity")
        st.markdown("---")
    st.write("## Helpful Resources:")
    st.markdown(
        """
        - [American Cancer Society](https://www.cancer.org)
        - [National Cancer Institute](https://www.cancer.gov)
        - [World Health Organization - Cancer](https://www.who.int/health-topics/cancer)
        - [Breast Cancer Research Foundation](https://www.bcrf.org)
        - [Cancer.net](https://www.cancer.net)
        """
    )
    st.markdown("<small>Stay informed with these trusted sources.</small>", unsafe_allow_html=True)

        
       
if __name__ == '__main__':
    main()