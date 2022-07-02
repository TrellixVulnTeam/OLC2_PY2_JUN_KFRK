#importar librerias
from unittest import main
import streamlit as st
import pickle
import pandas as pd

#extraer los archivos picke
with open('lin_reg.pkl','rb') as li:
    lin_reg = pickle.load(li)


#funcion para clasificar las plantas
def classify(num):
    if num == 0:
        return 'setosa'
    elif num == 1:
        return 'virginica'



def main():
    #titulo
    st.title('proyecto 2 compiladores')
    #titulo de sidebar
    st.sidebar.header('User input Parameters')

    #funcion para poner los parametros en el side bar
    def user_input_parameters():
        sepal_length = st.sidebar.slider('Sepal length',4.3,7.9,5.4)
        data = { 'sepal_length': sepal_length
                }
        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_parameters()

    #escoger el modelo preferido
    option = ['linear Regression','Regresion polinomial', 'Clasificador gaussiano','Arboles de descisión','Redes neuronales']
    model = st.sidebar.selectbox('Wich model you like to use?',option)

    st.subheader('User input Parameters')
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        if model == 'linear Regression':
            st.success(classify(lin_reg.predict(df)))
        

if __name__ == '__main__':
    main()