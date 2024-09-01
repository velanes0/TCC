import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import torch.nn.init as init
from torchmetrics import Precision, Recall, Accuracy
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from datetime import datetime


## Perceptron Multicamadas - Fully Connected Neural Netowrks
class MLP(nn.Module):
    
    """
    Classe que contem a implementação do modelo clássico do Perceptron Multicamadas (Multi Layer Perceptron)
    
    Funcionamento: 
    1) Para a primeira camada a recebe duas Int sendo o numero de inputs iniciais e o numero subsequete de neuronios 
    na primeira camada oculta --> nn.Linear(input_features,hidden_layer_1)
    2) Para melhorar o desempenho do modelo usa-se a normalização das bateladas, que consiste em aplicar uma normalização 
    dos pesos e vieses antes de passar pela camada da função de ativação
    3) A inicialização dos pesos segue a indicada por Kai-Ming (Referenciar artigo)
    ...

    Atributos
    ----------
    fc_i : Int
        Iésima camada linear de neuronios completamente conectados
    bn_i : Int
        Iésima camada de normalização de normalização de batelada

    Methods
    -------
    __init__
        Constroi e inicializa o modelo as camadas e  pesos e vieses iniciais  

    foward
        Aplica o passe para frente (Forward Pass) e retorna o neuronio final ativado com a classificacao para a observacao em questão

    """
    def __init__(self):
        super(MLP, self).__init__() 
        self.fc1 = nn.Linear(7, 50) # Definicao da primeira camada --> 7 variaveis, 50 Neuronios Ocultos
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 35)
        self.bn2 = nn.BatchNorm1d(35)
        self.fc3 = nn.Linear(35, 80)
        self.bn3 = nn.BatchNorm1d(80)
        self.fc4 = nn.Linear(80, 5)
        self.bn4 = nn.BatchNorm1d(5)
        self.fc5 = nn.Linear(5,3)  # Definicao da última camada --> 3 Classificacoes possiveis

        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)
        init.kaiming_uniform_(self.fc3.weight)
        init.kaiming_uniform_(self.fc4.weight)
        init.kaiming_uniform_(self.fc5.weight)
        
        
    def forward(self, x):
        x = nn.functional.elu( self.bn1(self.fc1(x)))
        x = nn.functional.elu( self.bn2(self.fc2(x)))
        x = nn.functional.elu( self.bn3(self.fc3(x)))
        x = nn.functional.elu( self.bn4(self.fc4(x)))
        x = self.fc5(x) # Para a ultima camada do forward pass não usa-se função de ativação (ja embutido na Loss Function)  
        return x
    


    
class CNN(nn.Module):
    """
    Classe que contem a implementação do modelo de Redes Neurais Convolucionais (Convolutional Neuron Networks)
    
    Funcionamento: 
    

    ...

    Atributos
    ----------
    num_classes : Int
        Numero de classes presentes (Sendo essas Operação Normal, Transicao para falha e Falha)
    

    Methods
    -------
    __init__
        Constroi e inicializa o modelo as camadas e  pesos e vieses iniciais  

    foward
        Aplica o passe para frente (Forward Pass) e retorna a classificacao para a observacao em questão

    _calculate_linear_input_size
        Calcula o tamanho final do size apos as camadas de convolucao    
    """
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        # Input é um tensor 1D ([batch_size, 1, 7]) tratando o feature tensor como contendo somente um canal
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # Convolucao em 1D
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # Convolucao em 1D
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        # Calculo do size depois de todas as camadas de convolução 
        self._to_linear = self._calculate_linear_input_size()

        # Classificador final
        self.classifier = nn.Linear(self._to_linear , num_classes) # Uso da variavel _to_linear para definir a conexao final entre os mapas e o numero de classes

    def forward(self, x):
        x = x.unsqueeze(1)  # Adicionando dimensao do canal de [batch_size, 7] para [batch_size, 1, 7] 
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def _calculate_linear_input_size(self):
        # Passe para frente com uma dummy variable para calcular o size final da rede
        with torch.no_grad():
            x = torch.randn(1, 7)  # Example input size (batch_size, number_of_features)
            x = x.unsqueeze(1)  # Adding channel dimension to match Conv1d input
            x = self.feature_extractor(x)
            return x.shape[1]


class REGULAR_RNN(nn.Module):

    """
    Classe que contem a implementação do modelo de Redes Neurais Recorrentes (Recurrent Neuron Networks)
    
    Funcionamento: 
    

    ...

    Atributos
    ----------
    num_classes : Int
        Numero de classes presentes (Sendo essas Operação Normal, Transicao para falha e Falha)
    

    Methods
    -------
    __init__
        Constroi e inicializa o modelo as camadas e  pesos e vieses iniciais  

    foward
        Aplica o passe para frente (Forward Pass) e retorna a classificacao para a observacao em questão
    
    """
    
    def __init__(self):
        super(REGULAR_RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=7,          # Numero de Input Features 
            hidden_size=32,        # Numero de unidades na camada oculta 
            num_layers=5,          # Numero de camadas recorrentes 
            batch_first=True)      # Dimensao da batelada como primeiro vetor do tensor
        self.fc = nn.Linear(32, 3) # Numero de neuronios conectados para  o output final 

    def forward(self, x):
        # Inicializacao dos pelos hidden state com zeros 
        h0 = torch.zeros(5, x.size(0), 32).to(x.device)

        # Passe e calculos dos valores iniciais da hidden state pela RNN 
        out, _ = self.rnn(x, h0)

        # Somente interessa a classificacao final que a rede gera, retorna-se o ultimo outpur do ultimo time step 
        out = self.fc(out[:, -1, :])

        return out



class RNN_LSTM(nn.Module):
    """
    Classe que contem a implementação do modelo de Redes Neurais Recorrentes (Recurrent Neuron Networks)
    
    Funcionamento: 
    

    ...

    Atributos
    ----------
    num_classes : Int
        Numero de classes presentes (Sendo essas Operação Normal, Transicao para falha e Falha)
    

    Methods
    -------
    __init__
        Constroi e inicializa o modelo as camadas e  pesos e vieses iniciais  

    foward
        Aplica o passe para frente (Forward Pass) e retorna a classificacao para a observacao em questão
    
    """
    def __init__(self):
        super(RNN_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=7,           # Numero de Input Features
            hidden_size=32,         # Numero de unidades na camada oculta 
            num_layers=5,           # Numero de camadas recorrentes 
            batch_first=True)       # Batch dimension is first
        self.fc = nn.Linear(32, 3)  # Numero de neuronios conectados para  o output final

    def forward(self, x):
        # Inicializacao dos pelos hidden state com zeros
        h0 = torch.zeros(5, x.size(0), 32).to(x.device)
        c0 = torch.zeros(5, x.size(0), 32).to(x.device)
        # Passe e calculos dos valores iniciais da hidden state pela RNN 
        out, _ = self.lstm(x, (h0,c0))

        # Somente interessa a classificacao final que a rede gera, retorna-se o ultimo outpur do ultimo time step
        out = self.fc(out[:, -1, :])

        return out
