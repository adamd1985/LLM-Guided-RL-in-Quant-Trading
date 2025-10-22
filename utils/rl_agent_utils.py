# coding=utf-8

"""
Goal: Implement a trading simulator to simulate and compare trading strategies.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""


###############################################################################
################################### Imports ###################################
###############################################################################

import copy
import numpy as np


shiftRange = [0]
stretchRange = [1]
filterRange = [5]
noiseRange = [0]

import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt

import math
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from abc import ABC, abstractmethod

import os
import sys
import importlib
import pickle
import itertools

import numpy as np
import pandas as pd

from tabulate import tabulate
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import os
import gymnasium as gym
import math
import numpy as np

import pandas as pd
pd.options.mode.chained_assignment = None

from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

# Boolean handling the saving of the stock market data downloaded
saving = True

# Default parameters related to the DQN algorithm
gamma = 0.4
learningRate = 0.0001
targetNetworkUpdate = 1000
learningUpdatePeriod = 1

# Default parameters related to the Experience Replay mechanism
capacity = 100000
batchSize = 32
experiencesRequired = 1000

# Default parameters related to the Deep Neural Network
numberOfNeurons = 512
dropout = 0.2

# Default parameters related to the Epsilon-Greedy exploration technique
epsilonStart = 1.0
epsilonEnd = 0.01
epsilonDecay = 10000

# Default parameters regarding the sticky actions RL generalization technique
alpha = 0.1

# Default parameters related to preprocessing
filterOrder = 5

# Default paramters related to the clipping of both the gradient and the RL rewards
gradientClipping = 1
rewardClipping = 1

# Default parameter related to the L2 Regularization
L2Factor = 0.000001

# Default paramter related to the hardware acceleration (CUDA)
GPUNumber = 0

# Variables defining the default trading horizon
startingDate = '2012-1-1'
endingDate = '2020-1-1'
splitingDate = '2018-1-1'

# Variables defining the default observation and state spaces
stateLength = 30
observationSpace = 1 + (stateLength - 1) * 6
actionSpace = 2

# Variables setting up the default transaction costs
percentageCosts = [0, 0.1, 0.2]
transactionCosts = percentageCosts[1]/100

# Variables specifying the default capital at the disposal of the trader
money = 100000

# Variables specifying the default general training parameters
bounds = [1, 30]
step = 1
numberOfEpisodes = 50

# Dictionary listing the fictive stocks supported
fictives = {
    'Linear Upward' : 'LINEARUP',
    'Linear Downward' : 'LINEARDOWN',
    'Sinusoidal' : 'SINUSOIDAL',
    'Triangle' : 'TRIANGLE',
}

 # Dictionary listing the 30 stocks considered as testbench
stocks = {
    'Dow Jones' : 'DIA',
    'S&P 500' : 'SPY',
    'NASDAQ 100' : 'QQQ',
    'FTSE 100' : 'EZU',
    'Nikkei 225' : 'EWJ',
    'Google' : 'GOOGL',
    'Apple' : 'AAPL',
    'Meta' : 'META',
    'Amazon' : 'AMZN',
    'Microsoft' : 'MSFT',
    'Twitter' : 'TWTR',
    'Nokia' : 'NOK',
    'Philips' : 'PHIA.AS',
    'Siemens' : 'SIE.DE',
    'Baidu' : 'BIDU',
    'Alibaba' : 'BABA',
    'Tencent' : '0700.HK',
    'Sony' : '6758.T',
    'JPMorgan Chase' : 'JPM',
    'HSBC' : 'HSBC',
    'CCB' : '0939.HK',
    'ExxonMobil' : 'XOM',
    'Shell' : 'RDSA.AS',
    'PetroChina' : 'PTR',
    'Tesla' : 'TSLA',
    'Volkswagen' : 'VWAGY',
    'Toyota' : 'TM',
    'Coca Cola' : 'KO',
    'AB InBev' : 'ABI.BR',
    'Kirin' : '2503.T'
}

# Dictionary listing the 5 trading indices considered as testbench
indices = {
    'Dow Jones' : 'DIA',
    'S&P 500' : 'SPY',
    'NASDAQ 100' : 'QQQ',
    'FTSE 100' : 'EZU',
    'Nikkei 225' : 'EWJ'
}

# Dictionary listing the 25 company stocks considered as testbench
companies = {
    'Google' : 'GOOGL',
    'Apple' : 'AAPL',
    'Facebook' : 'FB',
    'Amazon' : 'AMZN',
    'Microsoft' : 'MSFT',
    'Twitter' : 'TWTR',
    'Nokia' : 'NOK',
    'Philips' : 'PHIA.AS',
    'Siemens' : 'SIE.DE',
    'Baidu' : 'BIDU',
    'Alibaba' : 'BABA',
    'Tencent' : '0700.HK',
    'Sony' : '6758.T',
    'JPMorgan Chase' : 'JPM',
    'HSBC' : 'HSBC',
    'CCB' : '0939.HK',
    'ExxonMobil' : 'XOM',
    'Shell' : 'RDSA.AS',
    'PetroChina' : 'PTR',
    'Tesla' : 'TSLA',
    'Volkswagen' : 'VWAGY',
    'Toyota' : 'TM',
    'Coca Cola' : 'KO',
    'AB InBev' : 'ABI.BR',
    'Kirin' : '2503.T'
}

import pandas as pd
from matplotlib import pyplot as plt

import math
import random
import copy
import datetime

import numpy as np

from collections import deque
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter



import numpy as np
import pandas as pd
from scipy import signal


import os
import yfinance as yf
import pandas as pd
import requests

from io import StringIO


class DataAugmentation:
    """
    GOAL: Implementing some data augmentation techniques for stock time series.

    METHODS:
        - shiftTimeSeries: Generate a new trading environment by shifting the time series.
        - stretching: Generate a new trading environment by stretching or contracting the price series.
        - noiseAddition: Generate a new trading environment by adding noise to the time series.
        - lowPassFilter: Generate a new trading environment by applying a low-pass filter.
        - normalize: Normalize specific numerical columns.
        - processTradeAction: Process the trade_action column.
        - generate: Generate a set of new trading environments based on the augmentation techniques.
    """

    def shiftTimeSeries(self, tradingEnv, shiftMagnitude=0):
        """
        GOAL: Shift time series data for volume and specified features.

        INPUTS: - tradingEnv: Original trading environment to augment.
                - shiftMagnitude: Magnitude of the shift.

        OUTPUTS: - newTradingEnv: New trading environment generated.
        """
        newTradingEnv = copy.deepcopy(tradingEnv)

        if shiftMagnitude < 0:
            minValue = np.min(tradingEnv.data['Volume'])
            shiftMagnitude = max(-minValue, shiftMagnitude)

        newTradingEnv.data['Volume'] += shiftMagnitude

        return newTradingEnv

    def stretching(self, tradingEnv, factor=1):
        """
        GOAL: Stretch or contract the price time series by multiplying returns by a factor.

        INPUTS: - tradingEnv: Original trading environment to augment.
                - factor: Stretching/contraction factor.

        OUTPUTS: - newTradingEnv: New trading environment generated.
        """
        newTradingEnv = copy.deepcopy(tradingEnv)
        returns = newTradingEnv.data['Close'].pct_change() * factor

        for i in range(1, len(newTradingEnv.data.index)):
            newTradingEnv.data['Close'][i] = newTradingEnv.data['Close'][i-1] * (1 + returns[i])
            newTradingEnv.data['Low'][i] = newTradingEnv.data['Close'][i] * tradingEnv.data['Low'][i] / tradingEnv.data['Close'][i]
            newTradingEnv.data['High'][i] = newTradingEnv.data['Close'][i] * tradingEnv.data['High'][i] / tradingEnv.data['Close'][i]
            newTradingEnv.data['Open'][i] = newTradingEnv.data['Close'][i-1]

        # Stretch numerical features (entry_point, stop_loss, target) only at the point they were set!!!
        columns_to_stretch = ['entry_point', 'stop_loss', 'target']
        for col in columns_to_stretch:
            if col in newTradingEnv.data.columns:
                for i in range(1, len(newTradingEnv.data.index)):
                    if newTradingEnv.data[col][i] != newTradingEnv.data[col][i-1]:  # Value was set/changed at this point
                        original_close = tradingEnv.data['Close'][i]
                        newTradingEnv.data[col][i] = newTradingEnv.data[col][i] * (newTradingEnv.data['Close'][i] / original_close)

        return newTradingEnv

    def noiseAddition(self, tradingEnv, stdev=1):
        """
        GOAL: Add Gaussian random noise to the time series.

        INPUTS: - tradingEnv: Original trading environment to augment.
                - stdev: Standard deviation of the generated noise.

        OUTPUTS: - newTradingEnv: New trading environment generated.
        """
        newTradingEnv = copy.deepcopy(tradingEnv)

        for i in range(1, len(newTradingEnv.data.index)):
            price = newTradingEnv.data['Close'][i]
            volume = newTradingEnv.data['Volume'][i]
            priceNoise = np.random.normal(0, stdev * (price / 100))
            volumeNoise = np.random.normal(0, stdev * (volume / 100))

            newTradingEnv.data['Close'][i] *= (1 + priceNoise / 100)
            newTradingEnv.data['Low'][i] *= (1 + priceNoise / 100)
            newTradingEnv.data['High'][i] *= (1 + priceNoise / 100)
            newTradingEnv.data['Volume'][i] *= (1 + volumeNoise / 100)
            newTradingEnv.data['Open'][i] = newTradingEnv.data['Close'][i-1]

        return newTradingEnv

    def lowPassFilter(self, tradingEnv, order=5):
        """
        GOAL: Apply low-pass filter to smooth the time series.

        INPUTS: - tradingEnv: Original trading environment to augment.
                - order: Order of the low-pass filter.

        OUTPUTS: - newTradingEnv: New trading environment generated.
        """
        newTradingEnv = copy.deepcopy(tradingEnv)
        newTradingEnv.data['Close'] = newTradingEnv.data['Close'].rolling(window=order).mean()
        newTradingEnv.data['Low'] = newTradingEnv.data['Low'].rolling(window=order).mean()
        newTradingEnv.data['High'] = newTradingEnv.data['High'].rolling(window=order).mean()
        newTradingEnv.data['Volume'] = newTradingEnv.data['Volume'].rolling(window=order).mean()

        for i in range(order):
            newTradingEnv.data['Close'][i] = tradingEnv.data['Close'][i]
            newTradingEnv.data['Low'][i] = tradingEnv.data['Low'][i]
            newTradingEnv.data['High'][i] = tradingEnv.data['High'][i]
            newTradingEnv.data['Volume'][i] = tradingEnv.data['Volume'][i]

        newTradingEnv.data['Open'] = newTradingEnv.data['Close'].shift(1)
        newTradingEnv.data['Open'][0] = tradingEnv.data['Open'][0]

        return newTradingEnv



    def generate(self, tradingEnv, shiftRange=[0], stretchRange=[1], filterRange=[5], noiseRange=[0]):
        """
        Generate a set of new trading environments based on the data augmentation techniques.

        INPUTS: - tradingEnv: Original trading environment to augment.
                - shiftRange: Range of shifts to apply to the time series.
                - stretchRange: Range of stretching factors to apply.
                - filterRange: Range of low-pass filter orders.
                - noiseRange: Range of noise standard deviations.

        OUTPUTS: - tradingEnvList: List of augmented trading environments.
        """
        # return [tradingEnv]
        tradingEnvList = []
        for shift in shiftRange:
            tradingEnvShifted = self.shiftTimeSeries(tradingEnv, shift)
            for stretch in stretchRange:
                tradingEnvStretched = self.stretching(tradingEnvShifted, stretch)
                for order in filterRange:
                    tradingEnvFiltered = self.lowPassFilter(tradingEnvStretched, order)
                    for noise in noiseRange:
                        tradingEnvProcessed = self.noiseAddition(tradingEnvFiltered, noise)
                        tradingEnvList.append(tradingEnvProcessed)

        return tradingEnvList


###############################################################################
########################### Class YahooFinance ################################
###############################################################################


class YahooFinance:
    def __init__(self, data_folder='./data'):
        """
        Initializes the YahooFinance class.

        INPUTS:
        - data_folder: Path to the folder where cached data will be stored.
        """
        self.data = pd.DataFrame()
        self.data_folder = data_folder

        # Ensure the data folder exists
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

    def getDailyData(self, marketSymbol, startingDate, endingDate):
        """
        Retrieves daily data for a given stock symbol within the specified date range.
        If the data is cached, load it from the local file, otherwise download it.

        INPUTS:
        - marketSymbol: Stock market symbol.
        - startingDate: Start date for the data.
        - endingDate: End date for the data.

        OUTPUT:
        - DataFrame containing the daily stock data.
        """
        # Create a file name based on the market symbol and date range
        cache_file = os.path.join(self.data_folder, f"{marketSymbol}_{startingDate}_{endingDate}.csv")

        # If the file exists, load the data from cache
        if os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            self.data = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
        else:
            print(f"Downloading data for {marketSymbol} from {startingDate} to {endingDate}")
            # Download data from Yahoo Finance
            data = yf.download(marketSymbol, start=startingDate, end=endingDate)
            assert len(data) > 0
            self.data = self.processDataframe(data)

            # Save the downloaded data to the cache
            self.data.to_csv(cache_file)

        return self.data

    def processDataframe(self, dataframe):
        """
        Processes the Yahoo Finance dataframe by renaming columns and selecting relevant columns.

        INPUT:
        - dataframe: Raw dataframe from Yahoo Finance.

        OUTPUT:
        - Processed dataframe with selected columns.
        """
        dataframe['Close'] = dataframe['Adj Close']
        del dataframe['Adj Close']
        dataframe.index.names = ['Date']
        dataframe = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']]
        return dataframe


###############################################################################
############################### Class ReplayMemory ############################
###############################################################################

class ReplayMemory:
    """
    GOAL: Implementing the replay memory required for the Experience Replay
          mechanism of the DQN Reinforcement Learning algorithm.

    VARIABLES:  - memory: data structure storing the experiences.

    METHODS:    - __init__: Initialization of the memory data structure.
                - push: Insert a new experience into the replay memory.
                - sample: Sample a batch of experiences from the replay memory.
                - __len__: Return the length of the replay memory.
                - reset: Reset the replay memory.
    """

    def __init__(self, capacity=capacity):
        """
        GOAL: Initializating the replay memory data structure.

        INPUTS: - capacity: Capacity of the data structure, specifying the
                            maximum number of experiences to be stored
                            simultaneously.

        OUTPUTS: /
        """

        self.memory = deque(maxlen=capacity)


    def push(self, state, action, reward, nextState, done):
        """
        GOAL: Insert a new experience into the replay memory. An experience
              is composed of a state, an action, a reward, a next state and
              a termination signal.

        INPUTS: - state: RL state of the experience to be stored.
                - action: RL action of the experience to be stored.
                - reward: RL reward of the experience to be stored.
                - nextState: RL next state of the experience to be stored.
                - done: RL termination signal of the experience to be stored.

        OUTPUTS: /
        """

        self.memory.append((state, action, reward, nextState, done))


    def sample(self, batchSize):
        """
        GOAL: Sample a batch of experiences from the replay memory.

        INPUTS: - batchSize: Size of the batch to sample.

        OUTPUTS: - state: RL states of the experience batch sampled.
                 - action: RL actions of the experience batch sampled.
                 - reward: RL rewards of the experience batch sampled.
                 - nextState: RL next states of the experience batch sampled.
                 - done: RL termination signals of the experience batch sampled.
        """

        state, action, reward, nextState, done = zip(*random.sample(self.memory, batchSize))
        return state, action, reward, nextState, done


    def __len__(self):
        """
        GOAL: Return the capicity of the replay memory, which is the maximum number of
              experiences which can be simultaneously stored in the replay memory.

        INPUTS: /

        OUTPUTS: - length: Capacity of the replay memory.
        """

        return len(self.memory)


    def reset(self):
        """
        GOAL: Reset (empty) the replay memory.

        INPUTS: /

        OUTPUTS: /
        """

        self.memory = deque(maxlen=capacity)




###############################################################################
################################### Class DQN #################################
###############################################################################

class DQN(nn.Module):
    """
    GOAL: Implementing the Deep Neural Network of the DQN Reinforcement
          Learning algorithm.

    VARIABLES:  - fc1: Fully Connected layer number 1.
                - fc2: Fully Connected layer number 2.
                - fc3: Fully Connected layer number 3.
                - fc4: Fully Connected layer number 4.
                - fc5: Fully Connected layer number 5.
                - dropout1: Dropout layer number 1.
                - dropout2: Dropout layer number 2.
                - dropout3: Dropout layer number 3.
                - dropout4: Dropout layer number 4.
                - bn1: Batch normalization layer number 1.
                - bn2: Batch normalization layer number 2.
                - bn3: Batch normalization layer number 3.
                - bn4: Batch normalization layer number 4.

    METHODS:    - __init__: Initialization of the Deep Neural Network.
                - forward: Forward pass of the Deep Neural Network.
    """

    def __init__(self, numberOfInputs, numberOfOutputs, numberOfNeurons=numberOfNeurons, dropout=dropout):
        """
        GOAL: Defining and initializing the Deep Neural Network of the
              DQN Reinforcement Learning algorithm.

        INPUTS: - numberOfInputs: Number of inputs of the Deep Neural Network.
                - numberOfOutputs: Number of outputs of the Deep Neural Network.
                - numberOfNeurons: Number of neurons per layer in the Deep Neural Network.
                - dropout: Droupout probability value (handling of overfitting).

        OUTPUTS: /
        """

        # Call the constructor of the parent class (Pytorch torch.nn.Module)
        super(DQN, self).__init__()

        # Definition of some Fully Connected layers
        self.fc1 = nn.Linear(numberOfInputs, numberOfNeurons)
        self.fc2 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc3 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc4 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc5 = nn.Linear(numberOfNeurons, numberOfOutputs)

        # Definition of some Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(numberOfNeurons)
        self.bn2 = nn.BatchNorm1d(numberOfNeurons)
        self.bn3 = nn.BatchNorm1d(numberOfNeurons)
        self.bn4 = nn.BatchNorm1d(numberOfNeurons)

        # Definition of some Dropout layers.
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        # Xavier initialization for the entire neural network
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)


    def forward(self, input):
        """
        GOAL: Implementing the forward pass of the Deep Neural Network.

        INPUTS: - input: Input of the Deep Neural Network.

        OUTPUTS: - output: Output of the Deep Neural Network.
        """

        x = self.dropout1(F.leaky_relu(self.bn1(self.fc1(input))))
        x = self.dropout2(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.leaky_relu(self.bn4(self.fc4(x))))
        output = self.fc5(x)
        return output



###############################################################################
################################ Class TDQN ###################################
###############################################################################

class TDQN:
    """
    GOAL: Implementing an intelligent trading agent based on the DQN
          Reinforcement Learning algorithm.

    VARIABLES:  - device: Hardware specification (CPU or GPU).
                - gamma: Discount factor of the DQN algorithm.
                - learningRate: Learning rate of the ADAM optimizer.
                - capacity: Capacity of the experience replay memory.
                - batchSize: Size of the batch to sample from the replay memory.
                - targetNetworkUpdate: Frequency at which the target neural
                                       network is updated.
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - policyNetwork: Deep Neural Network representing the RL policy.
                - targetNetwork: Deep Neural Network representing a target
                                 for the policy Deep Neural Network.
                - optimizer: Deep Neural Network optimizer (ADAM).
                - replayMemory: Experience replay memory.
                - epsilonValue: Value of the Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - iterations: Counter of the number of iterations.

    METHODS:    - __init__: Initialization of the RL trading agent, by setting up
                            many variables and parameters.
                - getNormalizationCoefficients: Retrieve the coefficients required
                                                for the normalization of input data.
                - processState: Process the RL state received.
                - processReward: Clipping of the RL reward received.
                - updateTargetNetwork: Update the target network, by transfering
                                       the policy network parameters.
                - chooseAction: Choose a valid action based on the current state
                                observed, according to the RL policy learned.
                - chooseActionEpsilonGreedy: Choose a valid action based on the
                                             current state observed, according to
                                             the RL policy learned, following the
                                             Epsilon Greedy exploration mechanism.
                - learn: Sample a batch of experiences and learn from that info.
                - training: Train the trading DQN agent by interacting with its
                            trading environment.
                - testing: Test the DQN agent trading policy on a new trading environment.
                - plotExpectedPerformance: Plot the expected performance of the intelligent
                                   DRL trading agent.
                - saveModel: Save the RL policy model.
                - loadModel: Load the RL policy model.
                - plotTraining: Plot the training results (score evolution, etc.).
                - plotEpsilonAnnealing: Plot the annealing behaviour of the Epsilon
                                     (Epsilon-Greedy exploration technique).
    """

    def __init__(self, observationSpace, actionSpace, numberOfNeurons=numberOfNeurons, dropout=dropout,
                 gamma=gamma, learningRate=learningRate, targetNetworkUpdate=targetNetworkUpdate,
                 epsilonStart=epsilonStart, epsilonEnd=epsilonEnd, epsilonDecay=epsilonDecay,
                 capacity=capacity, batchSize=batchSize):
        """
        GOAL: Initializing the RL agent based on the DQN Reinforcement Learning
              algorithm, by setting up the DQN algorithm parameters as well as
              the DQN Deep Neural Network.

        INPUTS: - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - numberOfNeurons: Number of neurons per layer in the Deep Neural Network.
                - dropout: Droupout probability value (handling of overfitting).
                - gamma: Discount factor of the DQN algorithm.
                - learningRate: Learning rate of the ADAM optimizer.
                - targetNetworkUpdate: Update frequency of the target network.
                - epsilonStart: Initial (maximum) value of Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - epsilonEnd: Final (minimum) value of Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - epsilonDecay: Decay factor (exponential) of Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - capacity: Capacity of the Experience Replay memory.
                - batchSize: Size of the batch to sample from the replay memory.

        OUTPUTS: /
        """
        # Initialise the random function with a new random seed
        # random.seed(0)
        # np.random.seed(0)
        # torch.manual_seed(0)

        # Check availability of CUDA for the hardware (CPU or GPU)
        self.device = torch.device('cuda:'+str(GPUNumber) if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set the general parameters of the DQN algorithm
        self.gamma = gamma
        self.learningRate = learningRate
        self.targetNetworkUpdate = targetNetworkUpdate

        # Set the Experience Replay mechnism
        self.capacity = capacity
        self.batchSize = batchSize
        self.replayMemory = ReplayMemory(capacity)

        # Set both the observation and action spaces
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace

        # Set the two Deep Neural Networks of the DQN algorithm (policy and target)
        self.policyNetwork = DQN(observationSpace, actionSpace, numberOfNeurons, dropout).to(self.device)
        self.targetNetwork = DQN(observationSpace, actionSpace, numberOfNeurons, dropout).to(self.device)
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())
        self.policyNetwork.eval()
        self.targetNetwork.eval()

        # Set the Deep Learning optimizer
        self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=learningRate, weight_decay=L2Factor)

        # Set the Epsilon-Greedy exploration technique
        self.epsilonValue = lambda iteration: epsilonEnd + (epsilonStart - epsilonEnd) * math.exp(-1 * iteration / epsilonDecay)

        # Initialization of the iterations counter
        self.iterations = 0

        # Initialization of the tensorboard writer
        self.writer = SummaryWriter('runs/' + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S"))

    def getNormalizationCoefficients(self, tradingEnv):
        tradingData = tradingEnv.data
        closePrices = tradingData['Close'].tolist()
        lowPrices = tradingData['Low'].tolist()
        highPrices = tradingData['High'].tolist()
        volumes = tradingData['Volume'].tolist()
        trade_signal = tradingData['trade_signal'].tolist()

        coefficients = []
        margin = 1

        # 1. Close returns (abs)
        returns = [abs((closePrices[i] - closePrices[i - 1]) / closePrices[i - 1]) for i in range(1, len(closePrices))]
        coefficients.append((0, np.max(returns) * margin))

        # 2. High–Low delta
        deltaPrice = [abs(highPrices[i] - lowPrices[i]) for i in range(len(lowPrices))]
        coefficients.append((0, np.max(deltaPrice) * margin))

        # 3. Relative position (no normalization)
        coefficients.append((0, 1))

        # 4. Volume
        coefficients.append((np.min(volumes) / margin, np.max(volumes) * margin))

        # 5. trade_signal (min–max)
        coefficients.append((np.min(trade_signal) * margin, np.max(trade_signal) * margin))

        # 6. trade_action
        coefficients.append((0, 1))

        return coefficients

    def processState(self, state, coefficients):
        closePrices = state[0]
        lowPrices = state[1]
        highPrices = state[2]
        volumes = state[3]
        trade_signal = state[4]
        trade_action = state[5]

        # 1. Close returns
        returns = [(closePrices[i] - closePrices[i-1]) / closePrices[i-1] for i in range(1, len(closePrices))]
        state[0] = [(x - coefficients[0][0]) / (coefficients[0][1] - coefficients[0][0]) if coefficients[0][1] != coefficients[0][0] else 0 for x in returns]

        # 2. Low/High delta
        deltaPrice = [abs(highPrices[i] - lowPrices[i]) for i in range(1, len(lowPrices))]
        state[1] = [(x - coefficients[1][0]) / (coefficients[1][1] - coefficients[1][0]) if coefficients[1][1] != coefficients[1][0] else 0 for x in deltaPrice]

        # 3. Relative close position
        closePricePosition = []
        for i in range(1, len(closePrices)):
            d = abs(highPrices[i] - lowPrices[i])
            item = abs(closePrices[i] - lowPrices[i]) / d if d != 0 else 0.5
            closePricePosition.append(item)
        state[2] = [(x - coefficients[2][0]) / (coefficients[2][1] - coefficients[2][0]) if coefficients[2][1] != coefficients[2][0] else 0.5 for x in closePricePosition]

        # 4. Volume
        volumes = volumes[1:]
        state[3] = [(x - coefficients[3][0]) / (coefficients[3][1] - coefficients[3][0]) if coefficients[3][1] != coefficients[3][0] else 0 for x in volumes]

        # 5. Normalize trade_signal (state[4])
        trade_signal = trade_signal[1:]
        # state[4] = [(x - coefficients[4][0]) / (coefficients[4][1] - coefficients[4][0]) if coefficients[4][1] != coefficients[4][0] else 0 for x in trade_signal]
        state[4] = [x for x in trade_signal]

        # 6. Normalize trade_action (state[5])
        trade_action = trade_action[1:]
        state[5] = [x for x in trade_action]

        assert len(trade_signal) == len(trade_action)

        # 7. Compose final flattened state per time step
        state = [item for sublist in state for item in sublist]

        # Flatten into single vector
        return state

    def processReward(self, reward):
        """
        GOAL: Process the RL reward returned by the environment by clipping
              its value. Such technique has been shown to improve the stability
              the DQN algorithm.

        INPUTS: - reward: RL reward returned by the environment.

        OUTPUTS: - reward: Process RL reward.
        """

        return np.clip(reward, -rewardClipping, rewardClipping)


    def updateTargetNetwork(self):
        """
        GOAL: Taking into account the update frequency (parameter), update the
              target Deep Neural Network by copying the policy Deep Neural Network
              parameters (weights, bias, etc.).

        INPUTS: /

        OUTPUTS: /
        """

        # Check if an update is required (update frequency)
        if(self.iterations % targetNetworkUpdate == 0):
            # Transfer the DNN parameters (policy network -> target network)
            self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())


    def chooseAction(self, state):

        """
        GOAL: Choose a valid RL action from the action space according to the
              RL policy as well as the current RL state observed.

        INPUTS: - state: RL state returned by the environment.

        OUTPUTS: - action: RL action chosen from the action space.
                 - Q: State-action value function associated.
                 - QValues: Array of all the Qvalues outputted by the
                            Deep Neural Network.
        """

        # Choose the best action based on the RL policy
        with torch.no_grad():
            tensorState = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
            QValues = self.policyNetwork(tensorState).squeeze(0)
            Q, action = QValues.max(0)
            action = action.item()
            Q = Q.item()
            QValues = QValues.cpu().numpy()
            return action, Q, QValues


    def chooseActionEpsilonGreedy(self, state, previousAction):
        """
        GOAL: Choose a valid RL action from the action space according to the
              RL policy as well as the current RL state observed, following the
              Epsilon Greedy exploration mechanism.

        INPUTS: - state: RL state returned by the environment.
                - previousAction: Previous RL action executed by the agent.

        OUTPUTS: - action: RL action chosen from the action space.
                 - Q: State-action value function associated.
                 - QValues: Array of all the Qvalues outputted by the
                            Deep Neural Network.
        """

        # EXPLOITATION -> RL policy
        if(random.random() > self.epsilonValue(self.iterations)):
            # Sticky action (RL generalization mechanism)
            if(random.random() > alpha):
                action, Q, QValues = self.chooseAction(state)
            else:
                action = previousAction
                Q = 0
                QValues = [0, 0]

        # EXPLORATION -> Random
        else:
            action = random.randrange(self.actionSpace)
            Q = 0
            QValues = [0, 0]

        # Increment the iterations counter (for Epsilon Greedy)
        self.iterations += 1

        return action, Q, QValues


    def learning(self, batchSize=batchSize):
        """
        GOAL: Sample a batch of past experiences and learn from it
              by updating the Reinforcement Learning policy.

        INPUTS: batchSize: Size of the batch to sample from the replay memory.

        OUTPUTS: /
        """

        # Check that the replay memory is filled enough
        if (len(self.replayMemory) >= batchSize):

            # Set the Deep Neural Network in training mode
            self.policyNetwork.train()

            # Sample a batch of experiences from the replay memory
            state, action, reward, nextState, done = self.replayMemory.sample(batchSize)

            # Initialization of Pytorch tensors for the RL experience elements
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            action = torch.tensor(action, dtype=torch.long, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float, device=self.device)
            nextState = torch.tensor(nextState, dtype=torch.float, device=self.device)
            done = torch.tensor(done, dtype=torch.float, device=self.device)

            # Compute the current Q values returned by the policy network
            currentQValues = self.policyNetwork(state).gather(1, action.unsqueeze(1)).squeeze(1)

            # Compute the next Q values returned by the target network
            with torch.no_grad():
                nextActions = torch.max(self.policyNetwork(nextState), 1)[1]
                nextQValues = self.targetNetwork(nextState).gather(1, nextActions.unsqueeze(1)).squeeze(1)
                expectedQValues = reward + gamma * nextQValues * (1 - done)

            # Compute the Huber loss
            loss = F.smooth_l1_loss(currentQValues, expectedQValues)

            # Computation of the gradients
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), gradientClipping)

            # Perform the Deep Neural Network optimization
            self.optimizer.step()

            # If required, update the target deep neural network (update frequency)
            self.updateTargetNetwork()

            # Set back the Deep Neural Network in evaluation mode
            self.policyNetwork.eval()


    def training(self, trainingEnv, trainingParameters=[],
                 verbose=True, rendering=False, plotTraining=False, showPerformance=False):
        """
        GOAL: Train the RL trading agent by interacting with its trading environment.

        INPUTS: - trainingEnv: Training RL environment (known).
                - trainingParameters: Additional parameters associated
                                      with the training phase (e.g. the number
                                      of episodes).
                - verbose: Enable the printing of a training feedback.
                - rendering: Enable the training environment rendering.
                - plotTraining: Enable the plotting of the training results.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.

        OUTPUTS: - trainingEnv: Training RL environment.
        """

        """
        # Compute and plot the expected performance of the trading policy
        trainingEnv = self.plotExpectedPerformance(trainingEnv, trainingParameters, iterations=50)
        return trainingEnv
        """

        # Apply data augmentation techniques to improve the training set
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)

        # Initialization of some variables tracking the training and testing performances
        if plotTraining:
            # Training performance
            performanceTrain = []
            score = np.zeros((len(trainingEnvList), trainingParameters[0]))
            # Testing performance
            marketSymbol = trainingEnv.marketSymbol
            startingDate = trainingEnv.endingDate
            endingDate = '2020-01-01'
            money = trainingEnv.data['Money'][0]
            stateLength = trainingEnv.stateLength
            transactionCosts = trainingEnv.transactionCosts
            testingEnv = TradingEnv(trainingEnv.data_orig, startingDate, endingDate, money, stateLength, transactionCosts)
            performanceTest = []

        try:
            # If required, print the training progression
            if verbose:
                print("Training progression (hardware selected => " + str(self.device) + "):")

            # Training phase for the number of episodes specified as parameter
            for episode in tqdm(range(trainingParameters[0]), disable=not(verbose), desc=f"Training TQDM..", leave=False):

                # For each episode, train on the entire set of training environments
                for i in range(len(trainingEnvList)):

                    # Set the initial RL variables
                    coefficients = self.getNormalizationCoefficients(trainingEnvList[i])
                    trainingEnvList[i].reset()
                    startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                    trainingEnvList[i].setStartingPoint(startingPoint)
                    state = self.processState(trainingEnvList[i].state, coefficients)
                    assert all(isinstance(item, (int, float, np.number)) for item in state), f"State contains non-numeric values: {state}"

                    previousAction = 0
                    done = 0
                    stepsCounter = 0

                    # Set the performance tracking veriables
                    if plotTraining:
                        totalReward = 0

                    # Interact with the training environment until termination
                    while done == 0:

                        # Choose an action according to the RL policy and the current RL state
                        action, _, _ = self.chooseActionEpsilonGreedy(state, previousAction)

                        # Interact with the environment with the chosen action
                        nextState, reward, done, info = trainingEnvList[i].step(action)
                        # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                        reward = self.processReward(reward)
                        # Process the next state and push it to replay memory
                        nextState = self.processState(nextState, coefficients)
                        self.replayMemory.push(state, action, reward, nextState, done)

                        # Trick for better exploration
                        otherAction = int(not bool(action))
                        otherReward = self.processReward(info['reward'])
                        otherNextState = self.processState(info['State'], coefficients)
                        otherDone = info['Done']
                        self.replayMemory.push(state, otherAction, otherReward, otherNextState, otherDone)

                        # Execute the DQN learning procedure
                        stepsCounter += 1
                        if stepsCounter == learningUpdatePeriod:
                            self.learning()
                            stepsCounter = 0

                        # Update the RL state
                        state = nextState
                        previousAction = action

                        # Continuous tracking of the training performance
                        if plotTraining:
                            totalReward += reward

                    # Store the current training results
                    if plotTraining:
                        score[i][episode] = totalReward

                # Compute the current performance on both the training and testing sets
                if plotTraining:
                    # Training set performance
                    trainingEnv, q0, q1 = self.testing(trainingEnv, trainingEnv)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTrain.append(performance)
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', performance, episode)
                    trainingEnv.reset()
                    # Testing set performance
                    testingEnv, q0, q1 = self.testing(trainingEnv, testingEnv)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTest.append(performance)
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', performance, episode)
                    testingEnv.reset()

        except KeyboardInterrupt:
            print()
            print("WARNING: Training prematurely interrupted...")
            print()
            self.policyNetwork.eval()

        # Assess the algorithm performance on the training trading environment
        trainingEnv, q0, q1 = self.testing(trainingEnv, trainingEnv)

        # If required, show the rendering of the trading environment
        if rendering:
            trainingEnv.render()

        # If required, plot the training results
        if plotTraining:
            fig = plt.figure()
            ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
            ax.plot(performanceTrain)
            ax.plot(performanceTest)
            ax.legend(["Training", "Testing"])
            plt.savefig(''.join(['images/', str(marketSymbol), '_TrainingTestingPerformance', '.png']))
            #plt.show()
            for i in range(len(trainingEnvList)):
                self.plotTraining(score[i][:episode], marketSymbol)

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance('TDQN Train')

        # Closing of the tensorboard writer
        self.writer.close()

        return trainingEnv, q0, q1


    def testing(self, trainingEnv, testingEnv, rendering=False, showPerformance=False):
        """
        GOAL: Test the RL agent trading policy on a new trading environment
              in order to assess the trading strategy performance.

        INPUTS: - trainingEnv: Training RL environment (known).
                - testingEnv: Unknown trading RL environment.
                - rendering: Enable the trading environment rendering.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.

        OUTPUTS: - testingEnv: Trading environment backtested.
        """

        # Apply data augmentation techniques to process the testing set
        dataAugmentation = DataAugmentation()
        testingEnv.reset()
        testingEnvSmoothed = dataAugmentation.lowPassFilter(testingEnv, filterOrder)
        trainingEnv = dataAugmentation.lowPassFilter(trainingEnv, filterOrder)
        testingEnvSmoothed.reset()

        # Initialization of some RL variables
        coefficients = self.getNormalizationCoefficients(trainingEnv)
        state = self.processState(testingEnvSmoothed.state, coefficients)
        testingEnv.reset()
        QValues0 = []
        QValues1 = []
        done = 0

        # Interact with the environment until the episode termination
        while done == 0:

            # Choose an action according to the RL policy and the current RL state
            action, _, QValues = self.chooseAction(state)

            # Interact with the environment with the chosen action
            nextState, _, done, _ = testingEnvSmoothed.step(action)
            testingEnv.step(action)

            # Update the new state
            state = self.processState(nextState, coefficients)

            # Storing of the Q values
            QValues0.append(QValues[0])
            QValues1.append(QValues[1])

        # If required, show the rendering of the trading environment
        if rendering:
            testingEnv.render()
            self.plotQValues(QValues0, QValues1, testingEnv)

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance('TDQN Test')

        return testingEnv, QValues0, QValues0


    def plotTraining(self, score, marketSymbol):
        """
        GOAL: Plot the training phase results
              (score, sum of rewards).

        INPUTS: - score: Array of total episode rewards.
                - marketSymbol: Stock market trading symbol.

        OUTPUTS: /
        """

        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Total reward collected', xlabel='Episode')
        ax1.plot(score)
        plt.savefig(''.join(['images/', str(marketSymbol), 'TrainingResults', '.png']))
        #plt.show()


    def plotQValues(self, QValues0, QValues1, env):
        rewards = env.data['reward'].values

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Q values')
        ax1.plot(QValues0, label='Short Q', alpha=0.7)
        ax1.plot(QValues1, label='Long Q', alpha=0.7)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.set_ylabel('reward')
        ax2.plot(rewards, color='black', linewidth=1.0, label='reward', alpha=0.3)
        ax2.legend(loc='upper right')

        plt.title(f'{env.marketSymbol} - Q Values and Reward')
        plt.tight_layout()
        plt.savefig(f'images/{env.marketSymbol}_QValues_Reward.png')


    def plotExpectedPerformance(self, trainingEnv, trainingParameters=[], iterations=10):
        """
        GOAL: Plot the expected performance of the intelligent DRL trading agent.

        INPUTS: - trainingEnv: Training RL environment (known).
                - trainingParameters: Additional parameters associated
                                      with the training phase (e.g. the number
                                      of episodes).
                - iterations: Number of training/testing iterations to compute
                              the expected performance.

        OUTPUTS: - trainingEnv: Training RL environment.
        """

        # Preprocessing of the training set
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)

        # Save the initial Deep Neural Network weights
        initialWeights =  copy.deepcopy(self.policyNetwork.state_dict())

        # Initialization of some variables tracking both training and testing performances
        performanceTrain = np.zeros((trainingParameters[0], iterations))
        performanceTest = np.zeros((trainingParameters[0], iterations))

        # Initialization of the testing trading environment
        marketSymbol = trainingEnv.marketSymbol
        startingDate = trainingEnv.endingDate
        endingDate = '2020-1-1'
        money = trainingEnv.data['Money'][0]
        stateLength = trainingEnv.stateLength
        transactionCosts = trainingEnv.transactionCosts
        testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts)

        # Print the hardware selected for the training of the Deep Neural Network (either CPU or GPU)
        print("Hardware selected for training: " + str(self.device))

        try:

            # Apply the training/testing procedure for the number of iterations specified
            for iteration in range(iterations):

                # Print the progression
                print(''.join(["Expected performance evaluation progression: ", str(iteration+1), "/", str(iterations)]))

                # Training phase for the number of episodes specified as parameter
                for episode in tqdm(range(trainingParameters[0])):

                    # For each episode, train on the entire set of training environments
                    for i in range(len(trainingEnvList)):

                        # Set the initial RL variables
                        coefficients = self.getNormalizationCoefficients(trainingEnvList[i])
                        trainingEnvList[i].reset()
                        startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                        trainingEnvList[i].setStartingPoint(startingPoint)
                        state = self.processState(trainingEnvList[i].state, coefficients)
                        previousAction = 0
                        done = 0
                        stepsCounter = 0

                        # Interact with the training environment until termination
                        while done == 0:

                            # Choose an action according to the RL policy and the current RL state
                            action, _, _ = self.chooseActionEpsilonGreedy(state, previousAction)

                            # Interact with the environment with the chosen action
                            nextState, reward, done, info = trainingEnvList[i].step(action)

                            # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                            reward = self.processReward(reward)
                            nextState = self.processState(nextState, coefficients)
                            self.replayMemory.push(state, action, reward, nextState, done)

                            # Trick for better exploration
                            otherAction = int(not bool(action))
                            otherReward = self.processReward(info['reward'])
                            otherDone = info['Done']
                            otherNextState = self.processState(info['State'], coefficients)
                            self.replayMemory.push(state, otherAction, otherReward, otherNextState, otherDone)

                            # Execute the DQN learning procedure
                            stepsCounter += 1
                            if stepsCounter == learningUpdatePeriod:
                                self.learning()
                                stepsCounter = 0

                            # Update the RL state
                            state = nextState
                            previousAction = action

                    # Compute both training and testing  current performances
                    trainingEnv, q0, q1 = self.testing(trainingEnv, trainingEnv)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performanceTrain[episode][iteration] = analyser.computeSharpeRatio()
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', performanceTrain[episode][iteration], episode)
                    testingEnv, q0, q1 = self.testing(trainingEnv, testingEnv)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performanceTest[episode][iteration] = analyser.computeSharpeRatio()
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', performanceTest[episode][iteration], episode)

                # Restore the initial state of the intelligent RL agent
                if iteration < (iterations-1):
                    trainingEnv.reset()
                    testingEnv.reset()
                    self.policyNetwork.load_state_dict(initialWeights)
                    self.targetNetwork.load_state_dict(initialWeights)
                    self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=learningRate, weight_decay=L2Factor)
                    self.replayMemory.reset()
                    self.iterations = 0
                    stepsCounter = 0

            iteration += 1

        except KeyboardInterrupt:
            print()
            print("WARNING: Expected performance evaluation prematurely interrupted...")
            print()
            self.policyNetwork.eval()

        # Compute the expected performance of the intelligent DRL trading agent
        expectedPerformanceTrain = []
        expectedPerformanceTest = []
        stdPerformanceTrain = []
        stdPerformanceTest = []
        for episode in range(trainingParameters[0]):
            expectedPerformanceTrain.append(np.mean(performanceTrain[episode][:iteration]))
            expectedPerformanceTest.append(np.mean(performanceTest[episode][:iteration]))
            stdPerformanceTrain.append(np.std(performanceTrain[episode][:iteration]))
            stdPerformanceTest.append(np.std(performanceTest[episode][:iteration]))
        expectedPerformanceTrain = np.array(expectedPerformanceTrain)
        expectedPerformanceTest = np.array(expectedPerformanceTest)
        stdPerformanceTrain = np.array(stdPerformanceTrain)
        stdPerformanceTest = np.array(stdPerformanceTest)

        # Plot each training/testing iteration performance of the intelligent DRL trading agent
        for i in range(iteration):
            fig = plt.figure()
            ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
            ax.plot([performanceTrain[e][i] for e in range(trainingParameters[0])])
            ax.plot([performanceTest[e][i] for e in range(trainingParameters[0])])
            ax.legend(["Training", "Testing"])
            plt.savefig(''.join(['images/', str(marketSymbol), '_TrainingTestingPerformance', str(i+1), '.png']))
            #plt.show()

        # Plot the expected performance of the intelligent DRL trading agent
        fig = plt.figure()
        ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
        ax.plot(expectedPerformanceTrain)
        ax.plot(expectedPerformanceTest)
        ax.fill_between(range(len(expectedPerformanceTrain)), expectedPerformanceTrain-stdPerformanceTrain, expectedPerformanceTrain+stdPerformanceTrain, alpha=0.25)
        ax.fill_between(range(len(expectedPerformanceTest)), expectedPerformanceTest-stdPerformanceTest, expectedPerformanceTest+stdPerformanceTest, alpha=0.25)
        ax.legend(["Training", "Testing"])
        plt.savefig(''.join(['images/', str(marketSymbol), '_TrainingTestingExpectedPerformance', '.png']))
        #plt.show()

        # Closing of the tensorboard writer
        self.writer.close()

        return trainingEnv


    def saveModel(self, fileName):
        """
        GOAL: Save the RL policy, which is the policy Deep Neural Network.

        INPUTS: - fileName: Name of the file.

        OUTPUTS: /
        """

        torch.save(self.policyNetwork.state_dict(), fileName)


    def loadModel(self, fileName):
        """
        GOAL: Load a RL policy, which is the policy Deep Neural Network.

        INPUTS: - fileName: Name of the file.

        OUTPUTS: /
        """

        self.policyNetwork.load_state_dict(torch.load(fileName, map_location=self.device))
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())


    def plotEpsilonAnnealing(self):
        """
        GOAL: Plot the annealing behaviour of the Epsilon variable
              (Epsilon-Greedy exploration technique).

        INPUTS: /

        OUTPUTS: /
        """

        plt.figure()
        plt.plot([self.epsilonValue(i) for i in range(10*epsilonDecay)])
        plt.xlabel("Iterations")
        plt.ylabel("Epsilon value")
        plt.savefig(''.join(['images/', 'EpsilonAnnealing', '.png']))
        #plt.show()


###############################################################################
########################### Class tradingStrategy #############################
###############################################################################

class tradingStrategy(ABC):
    """
    GOAL: Define the abstract class representing a classical trading strategy.

    VARIABLES: /

    METHODS: - chooseAction: Make a decision regarding the next trading
                             position (long=1 and short=0).
             - training: Train the trading strategy on a known trading
                         environment (called training set) in order to
                         tune the trading strategy parameters.
             - testing: Test the trading strategy on another unknown trading
                        environment (called testing set) in order to evaluate
                        the trading strategy performance.
    """

    @abstractmethod
    def chooseAction(self, state):
        """
        GOAL: Make a decision regarding the next trading position
              (long=1 and short=0).

        INPUTS: - state: State of the trading environment.

        OUTPUTS: - action: Trading position decision (long=1 and short=0).
        """

        pass


    @abstractmethod
    def training(self, trainingEnv, trainingParameters=[],
                 verbose=False, rendering=False, plotTraining=False, showPerformance=False):
        """
        GOAL: Train the trading strategy on a known trading environment
              (called training set) in order to tune the trading strategy
              parameters.

        INPUTS: - trainingEnv: Known trading environment (training set).
                - trainingParameters: Additional parameters associated
                                      with the training phase.
                - verbose: Enable the printing of a training feedback.
                - rendering: Enable the trading environment rendering.
                - plotTraining: Enable the plotting of the training results.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.

        OUTPUTS: - trainingEnv: Trading environment associated with the best
                                trading strategy parameters backtested.
        """

        pass


    @abstractmethod
    def testing(self, testingEnv, trainingEnv, rendering=False, showPerformance=False):
        """
        GOAL: Test the trading strategy on another unknown trading
              environment (called testing set) in order to evaluate
              the trading strategy performance.

        INPUTS: - testingEnv: Unknown trading environment (testing set).
                - trainingEnv: Known trading environment (training set).
                - rendering: Enable the trading environment rendering.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.

        OUTPUTS: - testingEnv: Trading environment backtested.
        """

        pass



###############################################################################
########################### Class TradingSimulator ############################
###############################################################################

class TradingSimulator:
    """
    GOAL: Accurately simulating multiple trading strategies on different stocks
          to analyze and compare their performance.

    VARIABLES: /

    METHODS:   - displayTestbench: Display consecutively all the stocks
                                   included in the testbench.
               - analyseTimeSeries: Perform a detailled analysis of the stock
                                    market price time series.
               - plotEntireTrading: Plot the entire trading activity, with both
                                    the training and testing phases rendered on
                                    the same graph.
               - simulateNewStrategy: Simulate a new trading strategy on a
                                      a certain stock of the testbench.
               - simulateExistingStrategy: Simulate an already existing
                                           trading strategy on a certain
                                           stock of the testbench.
               - evaluateStrategy: Evaluate a trading strategy on the
                                   entire testbench.
               - evaluateStock: Compare different trading strategies
                                on a certain stock of the testbench.
    """

    def displayTestbench(self, startingDate=startingDate, endingDate=endingDate):
        """
        GOAL: Display consecutively all the stocks included in the
              testbench (trading indices and companies).

        INPUTS: - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.

        OUTPUTS: /
        """

        # Display the stocks included in the testbench (trading indices)
        for _, stock in indices.items():
            env = TradingEnv(stock, startingDate, endingDate, 0)
            env.render()

        # Display the stocks included in the testbench (companies)
        for _, stock in companies.items():
            env = TradingEnv(stock, startingDate, endingDate, 0)
            env.render()


    def plotEntireTrading(self, trainingEnv, testingEnv):
        """
        GOAL: Plot the entire trading activity, with both the training
              and testing phases rendered on the same graph for
              comparison purposes.

        INPUTS: - trainingEnv: Trading environment for training.
                - testingEnv: Trading environment for testing.

        OUTPUTS: /
        """

        # Artificial trick to assert the continuity of the Money curve
        ratio = trainingEnv.data['Money'][-1]/testingEnv.data['Money'][0]
        testingEnv.data['Money'] = ratio * testingEnv.data['Money']

        # Concatenation of the training and testing trading dataframes
        dataframes = [trainingEnv.data, testingEnv.data]
        data = pd.concat(dataframes)

        # Set the Matplotlib figure and subplots
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

        # Plot the first graph -> Evolution of the stock market price
        trainingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2)
        testingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2, label='_nolegend_')
        ax1.plot(data.loc[data['action'] == 1.0].index,
                 data['Close'][data['action'] == 1.0],
                 '^', markersize=5, color='green')
        ax1.plot(data.loc[data['action'] == -1.0].index,
                 data['Close'][data['action'] == -1.0],
                 'v', markersize=5, color='red')

        # Plot the second graph -> Evolution of the trading capital
        trainingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2)
        testingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2, label='_nolegend_')
        ax2.plot(data.loc[data['action'] == 1.0].index,
                 data['Money'][data['action'] == 1.0],
                 '^', markersize=5, color='green')
        ax2.plot(data.loc[data['action'] == -1.0].index,
                 data['Money'][data['action'] == -1.0],
                 'v', markersize=5, color='red')

        # Plot the vertical line seperating the training and testing datasets
        ax1.axvline(pd.to_datetime(splitingDate), color='black', linewidth=2.0)
        ax2.axvline(pd.to_datetime(splitingDate), color='black', linewidth=2.0)

        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long",  "Short", "Train/Test separation"])
        ax2.legend(["Capital", "Long", "Short", "Train/Test separation"])
        plt.savefig(''.join(['images/', str(trainingEnv.marketSymbol), '_TrainingTestingRendering', '.png']))
        #plt.show()


    def simulateNewStrategy(self,
                            stock_df,
                            ticker_symbol = None,
                            startingDate=startingDate,
                            endingDate=endingDate,
                            splitingDate=splitingDate,
                            observationSpace=observationSpace,
                            actionSpace=actionSpace,
                            money=money,
                            stateLength=stateLength,
                            transactionCosts=transactionCosts,
                            bounds=bounds,
                            step=step,
                            numberOfEpisodes=numberOfEpisodes,
                            verbose=True,
                            plotTraining=True,
                            rendering=True,
                            showPerformance=True,
                            models_path="./data/models/",
                            saveStrategy=False):
        """
        GOAL: Simulate a new trading strategy on a certain stock included in the
              testbench, with both learning and testing phases.

        INPUTS: - strategyName: Name of the trading strategy.
                - stockName: Name of the stock (in the testbench).
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splitingDate: Spliting date between the training dataset
                                and the testing dataset.
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - money: Initial capital at the disposal of the agent.
                - stateLength: Length of the trading agent state.
                - transactionCosts: Additional costs incurred while trading
                                    (e.g. 0.01 <=> 1% of transaction costs).
                - bounds: Bounds of the parameter search space (training).
                - step: Step of the parameter search space (training).
                - numberOfEpisodes: Number of epsiodes of the RL training phase.
                - verbose: Enable the printing of a simulation feedback.
                - plotTraining: Enable the plotting of the training results.
                - rendering: Enable the rendering of the trading environment.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
                - saveStrategy: Enable the saving of the trading strategy.

        OUTPUTS: - tradingStrategy: Trading strategy simulated.
                 - trainingEnv: Trading environment related to the training phase.
                 - testingEnv: Trading environment related to the testing phase.
        """

        # 1. INITIALIZATION PHASE
        trainingParameters = [numberOfEpisodes]

        # 2. TRAINING PHASE
        trainingEnv = TradingEnv(stock_df,
                                 startingDate,
                                 splitingDate,
                                 money,
                                 stateLength,
                                 transactionCosts)
        tradingStrategy = TDQN(observationSpace, actionSpace)

        trainingEnv, qt0, qt1 = tradingStrategy.training(trainingEnv,
                                               trainingParameters=trainingParameters,
                                               verbose=verbose,
                                               rendering=rendering,
                                               plotTraining=plotTraining,
                                               showPerformance=showPerformance)


        # 3. TESTING PHASE
        testingEnv = TradingEnv(stock_df,
                                splitingDate,
                                endingDate,
                                money,
                                stateLength,
                                transactionCosts)
        testingEnv, q0, q1 = tradingStrategy.testing(trainingEnv,
                                             testingEnv,
                                             rendering=rendering,
                                             showPerformance=showPerformance)
        if rendering:
            self.plotEntireTrading(trainingEnv, testingEnv)


        # 4. TERMINATION PHASE
        if(saveStrategy):
            fileName = "".join([models_path, "TQN", ticker_symbol, "_", startingDate, "_", splitingDate])
            tradingStrategy.saveModel(fileName)

        return tradingStrategy, trainingEnv, qt0, qt1, testingEnv, q0, q1

    def simulateExistingStrategy(self,
                                 stockName,
                                 startingDate=startingDate,
                                 endingDate=endingDate,
                                 splitingDate=splitingDate,
                                 observationSpace=observationSpace,
                                 actionSpace=actionSpace,
                                 money=money,
                                 stateLength=stateLength,
                                 transactionCosts=transactionCosts,
                                 rendering=True,
                                 showPerformance=True,
                                 models_path="./data/models/"):
        """
        GOAL: Simulate an already existing trading strategy on a certain
              stock of the testbench, the strategy being loaded from the
              strategy dataset. There is no training phase, only a testing
              phase.

        INPUTS: - strategyName: Name of the trading strategy.
                - stockName: Name of the stock (in the testbench).
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splitingDate: Spliting date between the training dataset
                                and the testing dataset.
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - money: Initial capital at the disposal of the agent.
                - stateLength: Length of the trading agent state.
                - transactionCosts: Additional costs incurred while trading
                                    (e.g. 0.01 <=> 1% of transaction costs).
                - rendering: Enable the rendering of the trading environment.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.

        OUTPUTS: - tradingStrategy: Trading strategy simulated.
                 - trainingEnv: Trading environment related to the training phase.
                 - testingEnv: Trading environment related to the testing phase.
        """
        # 1. INITIALIZATION PHASE
        if(stockName in indices):
            stock = indices[stockName]
        elif(stockName in companies):
            stock = companies[stockName]

        # 2. LOADING PHASE

        # Check that the strategy to load exists in the strategy dataset
        fileName = "".join([models_path, "DQN", stock, "_", startingDate, "_", splitingDate])
        exists = os.path.isfile(fileName)
        # If affirmative, load the trading strategy
        if exists:
            tradingStrategy = TDQN(observationSpace, actionSpace)
            tradingStrategy.loadModel(fileName)
        else:
            raise SystemError("The trading strategy specified does not exist, please provide a valid one.")


        # 3. TESTING PHASE
        trainingEnv = TradingEnv(stock,
                                 startingDate,
                                 splitingDate,
                                 money,
                                 stateLength,
                                 transactionCosts)
        testingEnv = TradingEnv(stock,
                                splitingDate,
                                endingDate,
                                money,
                                stateLength,
                                transactionCosts)

        # Testing of the trading strategy
        trainingEnv = tradingStrategy.testing(trainingEnv,
                                              trainingEnv,
                                              rendering=rendering,
                                              showPerformance=showPerformance)
        testingEnv = tradingStrategy.testing(trainingEnv,
                                             testingEnv,
                                             rendering=rendering,
                                             showPerformance=showPerformance)
        if rendering:
            self.plotEntireTrading(trainingEnv, testingEnv)

        return tradingStrategy, trainingEnv, testingEnv


###############################################################################
############################## Class TradingEnv ###############################
###############################################################################

class TradingEnv(gym.Env):
    """
    GOAL: Implement a custom trading environment compatible with Gymnasium.

    VARIABLES:  - data: Dataframe monitoring the trading activity.
                - state: RL state to be returned to the RL agent.
                - reward: RL reward to be returned to the RL agent.
                - done: RL episode termination signal.
                - t: Current trading time step.
                - marketSymbol: Stock market symbol.
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - stateLength: Number of trading time steps included in the state.
                - numberOfShares: Number of shares currently owned by the agent.
                - transactionCosts: Transaction costs associated with the trading
                                    activity (e.g. 0.01 is 1% of loss).

    METHODS:    - __init__: Object constructor initializing the trading environment.
                - reset: Perform a soft reset of the trading environment.
                - step: Transition to the next trading time step.
                - render: Illustrate graphically the trading environment.
    """

    def __init__(self,
                 stock_df,
                 startingDate,
                 endingDate,
                 money,
                 stateLength=30,
                 transactionCosts=0,
                 startingPoint=0,
                 ticker_symbol='TSLA'):
        """
        GOAL: Object constructor initializing the trading environment by setting up
              the trading activity dataframe as well as other important variables.

        INPUTS: - marketSymbol: Stock market symbol.
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - money: Initial amount of money at the disposal of the agent.
                - stateLength: Number of trading time steps included in the RL state.
                - transactionCosts: Transaction costs associated with the trading
                                    activity (e.g. 0.01 is 1% of loss).
                - startingPoint: Optional starting point (iteration) of the trading activity.

        OUTPUTS: /
        """
        startingDate = pd.to_datetime(startingDate, utc=True)
        endingDate = pd.to_datetime(endingDate, utc=True)
        self.data_orig = stock_df.copy()
        self.data = stock_df[(stock_df.index >= startingDate) & (stock_df.index < endingDate)].copy()
        if self.data.empty:
            raise ValueError(f"No data available between {startingDate} and {endingDate}")
        self.data.fillna(0, inplace=True)

        # Set the trading activity dataframe
        self.data['Position'] = 0
        self.data['action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['returns'] = 0.
        self.data['otherMoney'] = self.data['Holdings'] + self.data['Cash']
        self.data['otherReturns'] = 0.
        self.data['trade_align'] = 0.
        self.state = [
            self.data['Close'].iloc[:stateLength].tolist() if 'Close' in self.data.columns else [0] * stateLength,
            self.data['Low'].iloc[:stateLength].tolist() if 'Low' in self.data.columns else [0] * stateLength,
            self.data['High'].iloc[:stateLength].tolist() if 'High' in self.data.columns else [0] * stateLength,
            self.data['Volume'].iloc[:stateLength].tolist() if 'Volume' in self.data.columns else [0] * stateLength,
            self.data['trade_signal'].iloc[:stateLength].tolist() if 'trade_signal' in self.data.columns else [0] * stateLength,
            self.data['trade_action'].iloc[:stateLength].tolist() if 'trade_action' in self.data.columns else [0] * stateLength,
            [0] # Position
        ]

        self.reward = 0.
        self.done = 0

        # Set additional variables related to the trading activity
        self.marketSymbol = ticker_symbol
        self.startingDate = startingDate
        self.endingDate = endingDate
        self.stateLength = stateLength
        self.t = stateLength
        self.numberOfShares = 0
        self.transactionCosts = transactionCosts
        self.epsilon = 0.1

        # If required, set a custom starting point for the trading activity
        if startingPoint:
            self.setStartingPoint(startingPoint)


    def reset(self):
        """
        GOAL: Perform a soft reset of the trading environment.

        INPUTS: /

        OUTPUTS: - state: RL state returned to the trading strategy.
        """

        # Reset the trading activity dataframe
        self.data['Position'] = 0
        self.data['action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = self.data['Cash'][0]
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['returns'] = 0.

        self.data['otherReturns'] = 0.
        self.data['otherMoney'] = self.data['Holdings'] + self.data['Cash']
        self.data['reward'] = 0.
        # self.data['unshaped_reward'] = 0.
        self.data['other_reward'] = 0.
        # self.data['other_unshaped_reward'] = 0.

        # Reset the RL variables common to every Gymnasium environment
        # TODO: Features here:
        self.state = [
            self.data['Close'][0:self.stateLength].tolist(),
            self.data['Low'][0:self.stateLength].tolist(),
            self.data['High'][0:self.stateLength].tolist(),
            self.data['Volume'][0:self.stateLength].tolist(),
            self.data['trade_signal'][0:self.stateLength].tolist(),
            self.data['trade_action'][0:self.stateLength].tolist(),
            [0]  # Position
        ]

        self.reward = 0.
        self.done = 0

        # Reset additional variables related to the trading activity
        self.t = self.stateLength
        self.numberOfShares = 0

        return self.state


    def computeLowerBound(self, cash, numberOfShares, price):
        """
        GOAL: Compute the lower bound of the complete RL action space,
              i.e. the minimum number of share to trade.

        INPUTS: - cash: Value of the cash owned by the agent.
                - numberOfShares: Number of shares owned by the agent.
                - price: Last price observed.

        OUTPUTS: - lowerBound: Lower bound of the RL action space.
        """

        # Computation of the RL action lower bound
        deltaValues = - cash - numberOfShares * price * (1 + self.epsilon) * (1 + self.transactionCosts)
        if deltaValues < 0:
            lowerBound = deltaValues / (price * (2 * self.transactionCosts + (self.epsilon * (1 + self.transactionCosts))))
        else:
            lowerBound = deltaValues / (price * self.epsilon * (1 + self.transactionCosts))
        return lowerBound

    def shape_reward(self, t, raw_reward):
        trade_signal = self.data['trade_signal'][t]
        action = self.data['action'][t]
        llm_action = self.data['trade_action'][t]
        align = 1 if action == llm_action else -1

        shaped = raw_reward + (align * trade_signal)

        return shaped

    def step(self, action):
        """
        GOAL: Transition to the next trading time step based on the
              trading position decision made (either long or short).

        INPUTS: - action: Trading decision (1 = long, 0 = short).

        OUTPUTS: - state: RL state to be returned to the RL agent.
                 - reward: RL reward to be returned to the RL agent.
                 - done: RL episode termination signal (boolean).
                 - info: Additional information returned to the RL agent.
        """

        # Stting of some local variables
        t = self.t
        numberOfShares = self.numberOfShares
        customReward = False

        # CASE 1: LONG POSITION
        if(action == 1):
            self.data['Position'][t] = 1
            # Case a: Long -> Long
            if(self.data['Position'][t - 1] == 1):
                self.data['Cash'][t] = self.data['Cash'][t - 1]
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
            # Case b: No position -> Long
            elif(self.data['Position'][t - 1] == 0):
                self.numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
                self.data['action'][t] = 1
            # Case c: Short -> Long
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.numberOfShares = math.floor(self.data['Cash'][t]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
                self.data['action'][t] = 1

        # CASE 2: SHORT POSITION
        elif(action == 0):
            self.data['Position'][t] = -1
            # Case a: Short -> Short
            if(self.data['Position'][t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares, self.data['Close'][t-1])
                if lowerBound <= 0:
                    self.data['Cash'][t] = self.data['Cash'][t - 1]
                    self.data['Holdings'][t] =  - self.numberOfShares * self.data['Close'][t]
                else:
                    numberOfSharesToBuy = min(math.floor(lowerBound), self.numberOfShares)
                    self.numberOfShares -= numberOfSharesToBuy
                    self.data['Cash'][t] = self.data['Cash'][t - 1] - numberOfSharesToBuy * self.data['Close'][t] * (1 + self.transactionCosts)
                    self.data['Holdings'][t] =  - self.numberOfShares * self.data['Close'][t]
                    customReward = True
            # Case b: No position -> Short
            elif(self.data['Position'][t - 1] == 0):
                self.numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * self.data['Close'][t]
                self.data['action'][t] = -1
            # Case c: Long -> Short
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.numberOfShares = math.floor(self.data['Cash'][t]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * self.data['Close'][t]
                self.data['action'][t] = -1

        # CASE 3: PROHIBITED ACTION
        else:
            raise SystemExit("Prohibited action! Action should be either 1 (long) or 0 (short).")

        # Update the total amount of money owned by the agent, as well as the return generated
        self.data['Money'][t] = self.data['Holdings'][t] + self.data['Cash'][t]
        self.data['returns'][t] = (self.data['Money'][t] - self.data['Money'][t-1])/self.data['Money'][t-1]

        # Set the RL reward returned to the trading agent
        if not customReward:
            self.reward = self.data['returns'][t]
        else:
            self.reward = (self.data['Close'][t-1] - self.data['Close'][t])/self.data['Close'][t-1]
        self.reward = self.shape_reward(t, self.reward)
        self.data['reward'][t] = self.reward

        # Transition to the next trading time step
        # 'trade_action', 'entry_point', 'stop_loss', 'target'
        # TODO: FEATURES here:
        self.t = self.t + 1
        self.state = [
            self.data['Close'][self.t - self.stateLength : self.t].tolist(),
            self.data['Low'][self.t - self.stateLength : self.t].tolist(),
            self.data['High'][self.t - self.stateLength : self.t].tolist(),
            self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
            self.data['trade_signal'][self.t - self.stateLength : self.t].tolist(),
            self.data['trade_action'][self.t - self.stateLength : self.t].tolist(),
            [self.data['Position'][self.t - 1]],
        ]

        if(self.t == self.data.shape[0]):
            self.done = 1

        # Same reasoning with the other action (exploration trick)
        otherAction = int(not bool(action))
        customReward = False
        if(otherAction == 1):
            otherPosition = 1
            if(self.data['Position'][t - 1] == 1):
                otherCash = self.data['Cash'][t - 1]
                otherHoldings = numberOfShares * self.data['Close'][t]
            elif(self.data['Position'][t - 1] == 0):
                numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'][t - 1] - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                otherHoldings = numberOfShares * self.data['Close'][t]
            else:
                otherCash = self.data['Cash'][t - 1] - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                numberOfShares = math.floor(otherCash/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = otherCash - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                otherHoldings = numberOfShares * self.data['Close'][t]
        else:
            otherPosition = -1
            if(self.data['Position'][t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares, self.data['Close'][t-1])
                if lowerBound <= 0:
                    otherCash = self.data['Cash'][t - 1]
                    otherHoldings =  - numberOfShares * self.data['Close'][t]
                else:
                    numberOfSharesToBuy = min(math.floor(lowerBound), numberOfShares)
                    numberOfShares -= numberOfSharesToBuy
                    otherCash = self.data['Cash'][t - 1] - numberOfSharesToBuy * self.data['Close'][t] * (1 + self.transactionCosts)
                    otherHoldings =  - numberOfShares * self.data['Close'][t]
                    customReward = True
            elif(self.data['Position'][t - 1] == 0):
                numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'][t - 1] + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                otherHoldings = - numberOfShares * self.data['Close'][t]
            else:
                otherCash = self.data['Cash'][t - 1] + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                numberOfShares = math.floor(otherCash/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = otherCash + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                otherHoldings = - self.numberOfShares * self.data['Close'][t]

        otherMoney = otherHoldings + otherCash
        if not customReward:
            otherReward = (otherMoney - self.data['Money'][t-1])/self.data['Money'][t-1]
        else:
            otherReward = (self.data['Close'][t-1] - self.data['Close'][t])/self.data['Close'][t-1]

        otherReward = self.shape_reward(t, otherReward)
        self.data['other_reward'][t] = otherReward


        # TODO: FEATURES here:
        otherState = [
            self.data['Close'][self.t - self.stateLength : self.t].tolist(),
            self.data['Low'][self.t - self.stateLength : self.t].tolist(),
            self.data['High'][self.t - self.stateLength : self.t].tolist(),
            self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
            self.data['trade_signal'][self.t - self.stateLength : self.t].tolist(),
            self.data['trade_action'][self.t - self.stateLength : self.t].tolist(),
            [otherPosition]
        ]

        self.info = {'State' : otherState, 'reward' : otherReward, 'Done' : self.done}

        # Return the trading environment feedback to the RL trading agent
        return self.state, self.reward, self.done, self.info

    def render(self):
        """
        GOAL: Illustrate graphically the trading activity, by plotting
              both the evolution of the stock market price and the
              evolution of the trading capital. All the trading decisions
              (long and short positions) are displayed as well.

        INPUTS: /

        OUTPUTS: /
        """

        # Set the Matplotlib figure and subplots
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

        # Plot the first graph -> Evolution of the stock market price
        self.data['Close'].plot(ax=ax1, color='blue', lw=2)
        ax1.plot(self.data.loc[self.data['action'] == 1.0].index,
                 self.data['Close'][self.data['action'] == 1.0],
                 '^', markersize=5, color='green')
        ax1.plot(self.data.loc[self.data['action'] == -1.0].index,
                 self.data['Close'][self.data['action'] == -1.0],
                 'v', markersize=5, color='red')

        # Plot the second graph -> Evolution of the trading capital
        self.data['Money'].plot(ax=ax2, color='blue', lw=2)
        ax2.plot(self.data.loc[self.data['action'] == 1.0].index,
                 self.data['Money'][self.data['action'] == 1.0],
                 '^', markersize=5, color='green')
        ax2.plot(self.data.loc[self.data['action'] == -1.0].index,
                 self.data['Money'][self.data['action'] == -1.0],
                 'v', markersize=5, color='red')

        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long",  "Short"])
        ax2.legend(["Capital", "Long", "Short"])
        plt.savefig(''.join(['images/', str(self.marketSymbol), '_Rendering', '.png']))
        #plt.show()


    def setStartingPoint(self, startingPoint):
        """
        GOAL: Setting an arbitrary starting point regarding the trading activity.
              This technique is used for better generalization of the RL agent.

        INPUTS: - startingPoint: Optional starting point (iteration) of the trading activity.

        OUTPUTS: /
        """

        # Setting a custom starting point
        self.t = np.clip(startingPoint, self.stateLength, len(self.data.index))

        # Set the RL variables common to every Gymnasium environment
        # TODO: FEATURES here:
        self.state = [
            self.data['Close'][self.t - self.stateLength : self.t].tolist(),
            self.data['Low'][self.t - self.stateLength : self.t].tolist(),
            self.data['High'][self.t - self.stateLength : self.t].tolist(),
            self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
            self.data['trade_signal'][self.t - self.stateLength : self.t].tolist(),
            self.data['trade_action'][self.t - self.stateLength : self.t].tolist(),
            [self.data['Position'][self.t - 1]]
        ]

        if(self.t == self.data.shape[0]):
            self.done = 1

###############################################################################
######################### Class PerformanceEstimator ##########################
###############################################################################

class PerformanceEstimator:
    def __init__(self, tradingData, stateLength=30):
        self.data = tradingData
        self.stateLength = stateLength

    def computePnL(self):
        self.PnL = self.data["Money"][-1] - self.data["Money"][0]
        return self.PnL

    def computeCumulativeReturn(self):
        self.cumulativeReturn = ((1 + self.data['returns']).cumprod() - 1).iloc[-1]
        return self.cumulativeReturn

    def computeAnnualizedReturn(self):
        self.cumulativeReturn = self.computeCumulativeReturn()
        start = self.data.index[0].to_pydatetime()
        end = self.data.index[-1].to_pydatetime()
        timeElapsed = (end - start).days
        if self.cumulativeReturn > -1:
            self.annualizedReturn = 100 * (((1 + self.cumulativeReturn) ** (365 / timeElapsed)) - 1)
        else:
            self.annualizedReturn = -100.
        return self.annualizedReturn

    def computeAnnualizedVolatility(self):
        self.annualizedVolatily = 100 * np.sqrt(252) * self.data['returns'].std()
        return self.annualizedVolatily

    def computeSharpeRatio(self, riskFreeRate=0):
        expectedReturn = self.data['returns'].mean()
        volatility = self.data['returns'].std()
        if expectedReturn != 0 and volatility != 0:
            self.sharpeRatio = np.sqrt(252) * (expectedReturn - riskFreeRate) / volatility
        else:
            self.sharpeRatio = 0
        return self.sharpeRatio

    def computeSortinoRatio(self, riskFreeRate=0):
        expectedReturn = np.mean(self.data['returns'])
        negativeReturns = [r for r in self.data['returns'] if r < 0]
        volatility = np.std(negativeReturns)
        if expectedReturn != 0 and volatility != 0:
            self.sortinoRatio = np.sqrt(252) * (expectedReturn - riskFreeRate) / volatility
        else:
            self.sortinoRatio = 0
        return self.sortinoRatio

    def computeMaximumDrawdown(self):
        capital = self.data['Money'].values
        drawdowns = np.maximum.accumulate(capital) - capital
        peak = np.argmax(np.maximum.accumulate(capital) - capital)
        trough = np.argmax(capital[:peak]) if peak != 0 else 0
        self.maxDD = 100 * (capital[trough] - capital[peak]) / capital[trough] if trough != 0 else 0
        return self.maxDD

    def computeMeanDrawdownDuration(self):
        capital = self.data['Money'].values
        drawdowns = np.maximum.accumulate(capital) - capital
        is_in_drawdown = drawdowns > 0
        drawdown_durations = []
        duration = 0
        for val in is_in_drawdown:
            if val:
                duration += 1
            elif duration > 0:
                drawdown_durations.append(duration)
                duration = 0
        if duration > 0:
            drawdown_durations.append(duration)
        self.meanDrawdownDuration = np.mean(drawdown_durations) if drawdown_durations else 0
        return self.meanDrawdownDuration


    def computePortfolioTurnover(self):
        """
        GOAL: Compute the stricter Portfolio Turnover Ratio (PTR):
            PTR = (sum of absolute changes in holdings) / (average holdings over time).

        OUTPUT: Portfolio Turnover Ratio (PTR).
        """
        if 'Holdings' not in self.data:
            self.portfolioTurnover = 0
            return self.portfolioTurnover

        delta_holdings = self.data['Holdings'].diff().abs()
        total_trading_volume = delta_holdings.sum()
        average_holdings = self.data['Holdings'].mean()

        self.portfolioTurnover = total_trading_volume / average_holdings if average_holdings > 0 else 0
        return self.portfolioTurnover / 100.

    def computeProfitability(self):
        good, bad, profit, loss = 0, 0, 0, 0
        index = next((i for i in range(len(self.data.index)) if self.data['action'][i] != 0), None)
        if index is None:
            self.profitability = 0
            self.averageProfitLossRatio = 0
            return self.profitability, self.averageProfitLossRatio

        money = self.data['Money'][index]
        for i in range(index + 1, len(self.data.index)):
            if self.data['action'][i] != 0:
                delta = self.data['Money'][i] - money
                money = self.data['Money'][i]
                if delta >= 0:
                    good += 1
                    profit += delta
                else:
                    bad += 1
                    loss -= delta

        delta = self.data['Money'][-1] - money
        if delta >= 0:
            good += 1
            profit += delta
        else:
            bad += 1
            loss -= delta

        self.profitability = 100 * good / (good + bad)
        if good != 0:
            profit /= good
        if bad != 0:
            loss /= bad
        self.averageProfitLossRatio = 100 if loss == 0 else profit / loss
        return self.profitability, self.averageProfitLossRatio

    def computeSkewness(self):
        self.skewness = self.data['returns'].skew()
        return self.skewness

    def computePerformance(self):
        self.computePnL()
        self.computeAnnualizedReturn()
        self.computeAnnualizedVolatility()
        self.computeProfitability()
        self.computeSharpeRatio()
        self.computeSortinoRatio()
        self.computeMaximumDrawdown()
        self.computeMeanDrawdownDuration()
        self.computePortfolioTurnover()
        self.computeSkewness()

        self.performanceTable = [
            ["Profit & Loss (P&L)", f"{self.PnL:.0f}"],
            ["Cumulative Return", f"{self.cumulativeReturn:.2f}%"],
            ["Annualized Return", f"{self.annualizedReturn:.2f}%"],
            ["Annualized Volatility", f"{self.annualizedVolatily:.2f}%"],
            ["Sharpe Ratio", f"{self.sharpeRatio:.3f}"],
            ["Sortino Ratio", f"{self.sortinoRatio:.3f}"],
            ["Maximum Drawdown", f"{self.maxDD:.2f}%"],
            ["Mean Drawdown Duration", f"{self.meanDrawdownDuration:.2f} days"],
            ["Profitability", f"{self.profitability:.2f}%"],
            ["Ratio Average Profit/Loss", f"{self.averageProfitLossRatio:.3f}"],
            ["Portfolio Turnover", f"{self.portfolioTurnover:.3f}"],
            ["Skewness", f"{self.skewness:.3f}"],
        ]

        return self.performanceTable

    def displayPerformance(self, name):
        self.computePerformance()
        headers = ["Performance Indicator", name]
        tabulation = tabulate(self.performanceTable, headers, tablefmt="fancy_grid", stralign="center")
        print(tabulation)

    def getComputedPerformance(self):
        self.computePerformance()
        data = {
            'Metric': [
                "PnL", "Cumulative Return",
                "Annualized Return", "Annualized Volatility",
                "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
                "Mean Drawdown Duration", "Profitability",
                "Avg Profit/Loss Ratio", "Portfolio Turnover",
                "Skewness",
            ],
            'Value': [
                self.PnL, self.cumulativeReturn,
                self.annualizedReturn / 100, self.annualizedVolatily / 100,
                self.sharpeRatio, self.sortinoRatio, self.maxDD / 100,
                self.meanDrawdownDuration, self.profitability / 100,
                self.averageProfitLossRatio, self.portfolioTurnover,
                self.skewness,
            ]
        }
        return pd.DataFrame(data)
