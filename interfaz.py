#importar librerias
import os
from unittest import main
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
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
    st.sidebar.header('Selección del modelo a utilizar: ')

    #escoger el modelo preferido
    option = ['linear Regression','Regresion polinomial', 'Clasificador gaussiano','Arboles de descisión','Redes neuronales']
    model = st.sidebar.selectbox('Wich model you like to use?',option)

    st.subheader('Modelo seleccionado por el usuario: ')
    st.subheader(model)
    

    if model == 'linear Regression':
        FunLinearRegre()
    elif model == 'Regresion polinomial':
        FunRegrePol()
    elif model == 'Clasificador gaussiano':
        FunGaussiano()
    elif model == 'Arboles de descisión':
        FunArboles()
    elif model == 'Redes neuronales':
        FunRedes()
        #st.success(classify(lin_reg.predict(df)))

def FunLinearRegre():
    
    data = TypeArchi()
    if data is not None:
        
        df = pd.DataFrame(data)
        #obtener los parametros
        parameters1 = GetParameter1(df)
        parameters2 = GetParameter2(df)
        paramPredic = GetPrediction()
        new_min = 0
        new_max = 0


        if parameters1 != parameters2:
            # obtener la data
            x = data[parameters1].values.reshape((-1,1))
            y = data[parameters2]

            # se hace el entrenamiento del modelo
            regr = linear_model.LinearRegression()
            regr.fit(x,y)


            st.markdown('### Gráfica de dispersión')
            fig, ax = plt.subplots(1,1)
            fig.suptitle('Gráfica de Dispersión', fontsize="10")
            ax.grid()
            ax.set_xlabel(parameters1)
            ax.set_ylabel(parameters2)
            ax.scatter(x,y, color='black')
                    
            st.pyplot(fig)


            st.markdown('### Predicción:')
            if paramPredic is not None:
                new_min = 0
                new_max = int(paramPredic)
                x_new = np.linspace(new_min, new_max,50)
                x_new = x_new[:,np.newaxis]
                y_pred = regr.predict(x_new)
                st.info(max(y_pred))

            st.markdown('#### Coeficiente') 
            st.info(regr.coef_)
            st.write('Intercepto: ',regr.intercept_)
            st.markdown('### Función de tendencia: ')
            funStr = 'Y = '+ str(regr.coef_[0]) +"X"+" + "+str(regr.intercept_)
            st.info(funStr)
               
            st.table(data)
        else:
            st.warning('Los parametros deben de ser diferentes')
    

# metodo para obtener el parametro de la X
def GetParameter1(df:pd.DataFrame):
    options = df.columns.values
    selectionP = st.selectbox('Parametro 1 para el eje X', options)
    return selectionP

# metodo para obtener el parametro de la Y
def GetParameter2(df:pd.DataFrame):
    options = df.columns.values
    selectionP = st.selectbox('Parametro 2 para el eje Y', options)
    return selectionP


def GetPrediction():
    option = st.text_input('Ingrese la predicción que deseé obtener')
    return option
    

def GetGradePol():
    option = st.text_input('Ingrese el grado del polinomio')
    return option

def GetPrediGauss():
    option = st.text_input('Ingrese en forma de vector su prediccion')
    return option

def GetParameterGauss(df:pd.DataFrame):
    options = df.columns.values
    selectionP = st.multiselect('Seleccione las columnas para el modelo',options)
    return selectionP

def GetClassGauss(df:pd.DataFrame):
    options = df.columns.values
    selectionP = st.selectbox('Seleccione la clase', options)
    return selectionP

def GetOptionNeu():
    options = ['lbfgs','sgd','adam']
    selectionP = st.selectbox('seleccione el tipo de red: ', options)
    return selectionP


def TypeArchi():

    uploaded_file = st.file_uploader("Choose a file")
    data = None
    if uploaded_file is not None:
        try:
            split_tup = os.path.splitext(uploaded_file.name)
            # obtener la extension del archivo
            nombre_archi = split_tup[0]
            exte_archi = split_tup[1]
            if exte_archi == ".csv":
                data = pd.read_csv(uploaded_file)
            elif exte_archi == ".xls":
                data = pd.read_excel(upload_File)
            elif exte_archi ==".json":
                data = pd.read_json(upload_File)
            elif exte_archi == ".xlsx":
                data = pd.read_excel(uploaded_file)
        except:
            st.error("No se pudo cargar el archivo")

    return data
   


def FunRegrePol():
    data = TypeArchi()
    if data is not None:
       
        df = pd.DataFrame(data)
        #obtener los parametros
        parameters1 = GetParameter1(df)
        parameters2 = GetParameter2(df)
        paramPredic = GetPrediction()
        grade = GetGradePol()
        new_min = 0
        new_max = 0


        if parameters1 != parameters2 and grade != None:
            # obtener la data
            x = data[parameters1].values.reshape((-1,1))
            y = data[parameters2]

           

            
             # se hace el seteo del grado y la transformacion del eje x
            if grade != None:
                pol_features = PolynomialFeatures(degree=int(grade))
                x_trans = pol_features.fit_transform(x)
                
                st.markdown('### Gráfica de dispersión: ')
                fig, ax = plt.subplots(1,1)
                fig.suptitle('Gráfica de Dispersión', fontsize="10")
                ax.grid()
                ax.set_xlabel(parameters1)
                ax.set_ylabel(parameters2)
                ax.scatter(x,y, color='black')
                    
                st.pyplot(fig)


            # se hace el entrenamiento del modelo
            regr = linear_model.LinearRegression()
            regr.fit(x_trans,y)

            st.markdown('### Predicción:')
            if paramPredic != None:
                new_min = 0
                new_max = int(paramPredic)
                x_new = np.linspace(new_min, new_max,50)
                x_new = x_new[:,np.newaxis]
                # se hace el seteo del grado y la transformacion del eje x
                x_transc = pol_features.fit_transform(x_new)
                y_pred = regr.predict(x_transc)
                st.info(y_pred[y_pred.size-1])


            funStr = ""
            contador = 1
            if len(regr.coef_) >0:
                for item in regr.coef_[1:]:
                    if item != "0.0":
                        funStr += str(item)+"X"+"^"+str(contador) +"+"
                        contador = contador + 1

            funCom= "Y = " +funStr + str(regr.intercept_)
            st.markdown('### Función de tendencia: ')
            st.info(funCom)
            
            st.markdown('#### Coeficiente') 
            st.info(regr.coef_)
            st.markdown('#### Intercepto') 
            st.info(regr.intercept_)
            
            
            st.table(data)
        else:
            st.warning('Los parametros deben de ser diferentes')


def FunGaussiano():
    data = TypeArchi()
    if data is not  None:
        df = pd.DataFrame(data)

        # obtenemos la lista de parametros
        lstColumns = GetParameterGauss(df)
        columnClass = GetClassGauss(df)

       
        
        if len(lstColumns) > 0:
            try:
                st.write('Seleccione los datos para la predicción: ')
                cols = st.columns(len(lstColumns))
                ValoresPre = []
                contador = 0
                for item in lstColumns:
                    col = cols[contador]
                    ValoresPre.append(col.selectbox(str(item),data[item].unique(),0))
                    contador = contador + 1
                
                #utilizamos el encoder
                le = preprocessing.LabelEncoder()

                lstContent = []
                valoresEncoded = []
                contador2 = 0
                for pf in lstColumns:
                    Tupla = data[pf]
                    lstContent.append(Tupla)
                   
                
                nameClass = data[columnClass]

                
                lstEncondedP = []
                for item in lstContent:
                    Dupla = le.fit_transform(item)
                    lstEncondedP.append(Dupla)
                    valoresEncoded.append(np.where(le.classes_ == ValoresPre[contador2])[0][0])
                    contador2 = contador2 + 1

                
                playEncoded = le.fit_transform(nameClass)
                

                features = list(zip(*lstEncondedP))

                
                # Creamos clasificador gaussiano
                model = GaussianNB()
                # Entrenamos el modelo
                model.fit(features, playEncoded)

                predict = model.predict([valoresEncoded])
                predictClass = le.inverse_transform(predict)

                st.markdown('### Predicción: ')
                st.info(predictClass[0])

            except Exception as e:
                st.warning('error al cargar')       

        """
       """

    

def FunArboles():
    data = TypeArchi()
    if data is not None:
        df = pd.DataFrame(data)

        # obtenemos la lista de parametros
        lstColumns = GetParameterGauss(df)
        columnClass = GetClassGauss(df)


        if len(lstColumns) >0:
            lstContent = []
            for pf in lstColumns:
                Tupla = data[pf]
                lstContent.append(Tupla)
            
            nameClass = data[columnClass]

            #utilizamos el encoder
            le = preprocessing.LabelEncoder()
            lstEncondedP = []
            for item in lstContent:
                Dupla = le.fit_transform(item)
                lstEncondedP.append(Dupla)

            playEncoded = le.fit_transform(nameClass)

            features = list(zip(*lstEncondedP))

            model = DecisionTreeClassifier().fit(features, playEncoded)
            st.markdown('### Gráfica del árbol: ')
            plot_tree(model, filled=True)
            plt.savefig('arbol.png')
            #plt.close()
            
            image4 = Image.open('arbol.png')
            st.image(image4,width=1200,use_column_width='auto')
        else:
            st.warning('Debe seleccionar por lo menos una columna para realizar el algoritmo')
               

def FunRedes():
    data = TypeArchi()
    if data is not None:
        # leemos el archivo
        df = pd.DataFrame(data)

        # obtenemos la lista de parametros
        lstColumns = GetParameterGauss(df)
        columnClass = GetClassGauss(df)
        tipoNeu = GetOptionNeu()
        
        if len(lstColumns) > 0:
            try:
                st.write('Seleccione los datos para la predicción: ')
                cols = st.columns(len(lstColumns))
                ValoresPre = []
                contador = 0
                for item in lstColumns:
                    col = cols[contador]
                    ValoresPre.append(col.selectbox(str(item),data[item].unique(),0))
                    contador = contador + 1
                
                #utilizamos el encoder
                le = preprocessing.LabelEncoder()

                lstContent = []
                valoresEncoded = []
                contador2 = 0
                for pf in lstColumns:
                    Tupla = data[pf]
                    lstContent.append(Tupla)
                   
                
                nameClass = data[columnClass]

                
                lstEncondedP = []
                for item in lstContent:
                    Dupla = le.fit_transform(item)
                    lstEncondedP.append(Dupla)
                    valoresEncoded.append(np.where(le.classes_ == ValoresPre[contador2])[0][0])
                    contador2 = contador2 + 1

                
                playEncoded = le.fit_transform(nameClass)
                

                features = list(zip(*lstEncondedP))

                model = MLPClassifier(solver=tipoNeu, max_iter=500, hidden_layer_sizes=(100,100,100), random_state=0, tol=0.000001, verbose=10).fit(features, playEncoded)
                prediction = model.predict([valoresEncoded])
                predictionClass = le.inverse_transform(prediction)

                st.markdown('### Prediccion')
                st.info(predictionClass[0])
            except Exception as e:
                st.warning('error al cargar')

        else:
            st.warning('Debe seleccionar por lo menos una columna para realizar el algoritmo')

if __name__ == '__main__':
    main()