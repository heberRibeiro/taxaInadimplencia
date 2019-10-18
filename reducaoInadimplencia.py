import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Carregamento base de dados
dadosCredito = pd.read_csv('Credit.csv')

# Identificação dos atributos categóricos (tipo 'Object')
atributosParaEncoder = []
for i in list(dadosCredito.columns):
    if(dadosCredito[i].dtype == 'O'):
        atributosParaEncoder.append(i)
del i

# Remoção do atributo "class" da lista de atributos para o encoder
atributosParaEncoder.remove('class')

# Encoder dos atributos do tipo 'Object' para o modelo      
labelencoder = LabelEncoder()
for i in atributosParaEncoder:
    dadosCredito[i] = labelencoder.fit_transform(
            dadosCredito[i])
del i
  
# Definição dos atributos previsores e do atributo da classe
previsores = dadosCredito.iloc[:, 0:20].values
classe = dadosCredito.iloc[:, 20].values

# Formação da base de dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(previsores, classe, 
                                                    test_size = 0.3, 
                                                    random_state = 0)

# Treinamento do modelo
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# Teste do modelo
previsoes = naive_bayes.predict(X_test)
confusao = confusion_matrix(y_test, previsoes)
taxaAcerto = accuracy_score(y_test, previsoes)
taxaErro = 1 - taxaAcerto

from yellowbrick.classifier import ConfusionMatrix
visualizador = ConfusionMatrix(GaussianNB())
visualizador.fit(X_train, y_train)
visualizador.score(X_test, y_test)
visualizador.poof

# Taxa de inadimplência como resultado do teste: 20,7%
falsoPositivo = confusao[0,1]
verdadeiraPositivo = confusao[1,1]
taxaInadimplencia = falsoPositivo / (falsoPositivo + verdadeiraPositivo)