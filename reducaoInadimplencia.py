import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Carregamento base de dados
dadosCredito = pd.read_csv('Credito.csv', sep=';', encoding = 'cp860')

# Identificação dos atributos categóricos (tipo 'Object')
atributosParaEncoder = []
for i in list(dadosCredito.columns):
    if(dadosCredito[i].dtype == 'O'):
        atributosParaEncoder.append(i)
del i

# Remoção do atributo "class" da lista de atributos para o encoder
atributosParaEncoder.remove('CLASSE')

# Encoder dos atributos do tipo 'Object' para o modelo      
labelencoder = LabelEncoder()
for i in atributosParaEncoder:
    dadosCredito[i] = labelencoder.fit_transform(
            dadosCredito[i])
del i
  
# Definição dos atributos previsores e do atributo da classe
previsores = dadosCredito.iloc[:, 0:19].values
classe = dadosCredito.iloc[:, 19].values

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

# Intâncias nos dados de teste classificados como "bom": 214, "ruim": 86
# Diante dessa informação, identifica-se que as linhas da visualização
# correspodem aos dados de teste, e as colunas aos dados de previsão.
df_y_test = pd.DataFrame(y_test, columns=['classe'])
df_y_test[df_y_test['classe']=='bom'].count() #214
df_y_test[df_y_test['classe']=='ruim'].count() #86

# Taxa de inadimplência como resultado do teste: 20,1%
falsoPositivo = confusao[1,0]
verdadeiraPositivo = confusao[0,0]
taxaInadimplencia = falsoPositivo / (falsoPositivo + verdadeiraPositivo)