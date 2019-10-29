
# coding: utf-8

# # Funcion de preprocesamiento 

# In[3]:
import os
import timeit
import re
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from PyPDF2 import PdfFileReader    
from sklearn.feature_extraction.text import CountVectorizer

a= 'hola'

# In[37]:


def preprocesamiento(path,modelo = 'All', info='TRUE'):
        
    if (info == 'TRUE'):
        print('''
        
        1.-Primer argumento de la funcion; 
        
        ¿Dónde están los PDFs que queremos clasificar?
        
        
        2.-Segundo argumento, elección del modelo:
        
           * Mostrar todos los modelos -> 'All'
           
           * K-Nearest Neighbor        -> 'KNN'
           
           * Logistic Regression       -> 'LR'
           
           * Naive Bayes               -> 'NB'
           
        3.- Mostrar información.
        
        ''')
    else:
        pass
    
    # Creación de las listas

    List_Documents=[]
    ListDirDocuments=[]
    NumberOfPagesList=[]
    TextExtractionList=[]
    TEL = []
    TELstr=[] 
  
    
    # Directorios y nombres de documentos

    for (root, directories, filenames) in os.walk(path):
        for f in filenames:
            List_Documents.append(os.path.join(f))
            List_Documents=list(filter(lambda x: (x.endswith('pdf')), List_Documents))
            ListDirDocuments.append(os.path.join(root,f)#.replace('\\','/'))
            ListDirDocuments=list(filter(lambda x: (x.endswith('pdf')), ListDirDocuments))
  
    
   # Número de páginas de cada archivo
    
    for i in range(0,len(ListDirDocuments)):
    
        NumberOfPagesList.append(PdfFileReader(stream=ListDirDocuments[i]).getNumPages()) 
       
    
    df1=pd.DataFrame({'Nombre_Documento':List_Documents,
                      'Ubicación':ListDirDocuments,
                      'Número de Páginas':NumberOfPagesList})
    
    df1=df1[['Nombre_Documento','Número de Páginas','Ubicación']]
    

    # Extracción de texto
    
    for i in range(0,len(ListDirDocuments)):
    
        read_pdf = PdfFileReader(ListDirDocuments[i])
        number_of_pages = read_pdf.getNumPages()
        
        for page_number in range(number_of_pages):
            page = read_pdf.getPage(page_number)
            page_content = page.extractText()
            TextExtractionList.append(page_content)
  
    # Redimensionador
    
    a = 0
    for np in NumberOfPagesList:       
        LP = TextExtractionList[a:a+np]
        a+= np
        TEL.append(LP)
    

    for i in range(len(TEL)):
        TELstr.append(" ".join(TEL[i]))
        
        
    df2=pd.DataFrame({'Documento':List_Documents,
                  'Num.Páginas':NumberOfPagesList,
                  'AllRawText':TELstr,
                  'Dir':ListDirDocuments})
        
    #Limpieza de texto
    
    import timeit
    start_time = timeit.default_timer()

    from nltk.stem import SnowballStemmer
    stemmer=SnowballStemmer("english")
    from nltk.corpus import stopwords
    WordsStoped=stopwords.words("english")

    import re
    df2['Texto+Limpio']=df2.AllRawText.apply    (lambda x: " ".join([stemmer.stem(i) for i in re.sub('[^a-zA-Z]'," ",x).split() if i not in WordsStoped]).lower())

    df3 = df2.drop('Dir',axis=1)
    
    
    # Vectorizar y pasar el modelo
    
    from sklearn.feature_extraction.text import CountVectorizer
    
    vect = joblib.load('vectorizer.pkl')

    prueba =  vect.transform(df3['Texto+Limpio'])
    
    if (modelo == 'NB'):
        
        nb = joblib.load('modelos/modeloNB.pkl')
        
        resultado = nb.predict(prueba)
        
    elif (modelo =='LR'):

        lr = joblib.load('modelos/modeloRL.pkl')
        
        resultado = lr.predict(prueba)
        
    elif (modelo =='KNN'):

        knn = joblib.load('modelos/modeloKNN.pkl')
        
        resultado = knn.predict(prueba)
   
    elif (modelo == 'All'):
        
        nb = joblib.load('modelos/modeloNB.pkl')
        lr = joblib.load('modelos/modeloRL.pkl')
        knn = joblib.load('modelos/modeloKNN.pkl')
        
        resNb = nb.predict(prueba)
        resRl = lr.predict(prueba)
        resKnn = knn.predict(prueba)
    
    # Respuesta
    
    if (modelo != 'All'):
        
        res = pd.DataFrame({'Environmental': [len(resultado[resultado=='ENVIRONMENTAL'])]
                            ,'Technical':[len(resultado[resultado=='TECHNICAL'])]
                            ,'Social':[len(resultado[resultado=='SOCIAL'])]
                            ,'Broad Approvals':[len(resultado[resultado=='BROAD APPROVALS'])]
                            ,'Financial':[len(resultado[resultado=='FINANCIAL'])]})
    else:
        
        data = {'K-Nearest Neighbors': [len(resNb[resNb=='ENVIRONMENTAL'])
                                        ,len(resNb[resNb=='TECHNICAL'])
                                        ,len(resNb[resNb=='SOCIAL'])
                                        ,len(resNb[resNb=='BROAD APPROVALS'])
                                        ,len(resNb[resNb=='FINANCIAL'])]
                ,'Logistic Regression': [len(resRl[resRl=='ENVIRONMENTAL'])
                                         ,len(resRl[resRl=='TECHNICAL'])
                                         ,len(resRl[resRl=='SOCIAL'])
                                         ,len(resRl[resRl=='BROAD APPROVALS'])
                                         ,len(resRl[resRl=='FINANCIAL'])]
                         ,'Naive Bayes': [len(resKnn[resKnn=='ENVIRONMENTAL'])
                                         ,len(resKnn[resKnn=='TECHNICAL'])
                                         ,len(resKnn[resKnn=='SOCIAL'])
                                         ,len(resKnn[resKnn=='BROAD APPROVALS'])
                                         ,len(resKnn[resKnn=='FINANCIAL'])]}
        
        res = pd.DataFrame(data, index = ['ENVIRONMENTAL', 'TECHNICAL', 'SOCIAL', 'BROAD APPROVALS', 'FINANCIAL'])                  

    return(res)


# In[38]:


preprocesamiento('/Users/oscarcameaneddine/Desktop/llamada/PDF', 'All')

