import pandas as pd
import numpy as np


def load(file, attributes):
    df = pd.io.parsers.read_csv(file, header=None, usecols=np.arange(len(attributes)))
    df.columns= attributes
    return df


    #@mfunction("TrainingTime, TrainingAccuracy")
def elm_train(TrainingData_File=None, attributes=None, Elm_Type=None, NumberofHiddenNeurons=None, ActivationFunction=None):

    # Usage: elm_train(TrainingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
    # OR:    [TrainingTime, TrainingAccuracy] = elm_train(TrainingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
    #
    # Input:
    # TrainingData_File     - Filename of training data set
    # Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
    # NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
    # ActivationFunction    - Type of activation function:
    #                           'sig' for Sigmoidal function
    #                           'sin' for Sine function
    #                           'hardlim' for Hardlim function
    #
    # Output: 
    # TrainingTime          - Time (seconds) spent on training ELM
    # TrainingAccuracy      - Training accuracy: 
    #                           RMSE for regression or correct classification rate for classification
    #
    # MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
    # FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
    # neurons; neuron 5 has the highest output means input belongs to 5-th class
    #
    # Sample1 regression: [TrainingTime, TrainingAccuracy, TestingAccuracy] = elm_train('sinc_train', 0, 20, 'sig')
    # Sample2 classification: elm_train('diabetes_train', 1, 20, 'sig')
    #
    #%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    #%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    #%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    #%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    #%%%    DATE:       APRIL 2004

    #%%%%%%%%%% Macro definition
    REGRESSION = 0
    CLASSIFIER = 1

    #%%%%%%%%%% Load training dataset
    train_data = load(TrainingData_File, attributes)
    T = train_data(mslice[:], 1).cT
    P = train_data(mslice[:], mslice[2:size(train_data, 2)]).cT
    clear(mstring('train_data'))#   Release raw training data array

    NumberofTrainingData = size(P, 2)
    NumberofInputNeurons = size(P, 1)

    if Elm_Type != REGRESSION:
        #%%%%%%%%%%% Preprocessing the data of classification
        sorted_target = sort(T, 2)
        label = zeros(1, 1)    #   Find and save in 'label' class label from training and testing data sets
        label(1, 1).lvalue = sorted_target(1, 1)
        j = 1
        for i in mslice[2:NumberofTrainingData]:
            if sorted_target(1, i) != label(1, j):
                j = j + 1
                label(1, j).lvalue = sorted_target(1, i)
            end
        end
        number_class = j
        NumberofOutputNeurons = number_class

        #%%%%%%%%% Processing the targets of training
        temp_T = zeros(NumberofOutputNeurons, NumberofTrainingData)
        for i in mslice[1:NumberofTrainingData]:
            for j in mslice[1:number_class]:
                if label(1, j) == T(1, i):
                    break
                end
            end
            temp_T(j, i).lvalue = 1
        end
        T = temp_T * 2 - 1
    end#   end if of Elm_Type

    #%%%%%%%%%% Calculate weights & biases
    start_time_train = cputime

    #%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
    InputWeight = rand(NumberofHiddenNeurons, NumberofInputNeurons) * 2 - 1
    BiasofHiddenNeurons = rand(NumberofHiddenNeurons, 1)
    tempH = InputWeight * P
    clear(mstring('P'))#   Release input of training data
    ind = ones(1, NumberofTrainingData)
    BiasMatrix = BiasofHiddenNeurons(mslice[:], ind)#   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
    tempH = tempH + BiasMatrix

    #%%%%%%%%%% Calculate hidden neuron output matrix H
    __switch_0__ = lower(ActivationFunction)
    if 0:
        pass
    elif __switch_0__ == mcellarray([mstring('sig'), mstring('sigmoid')]):
        #%%%%%%% Sigmoid 
        H = 1 /eldiv/ (1 + exp(-tempH))
    elif __switch_0__ == mcellarray([mstring('sin'), mstring('sine')]):
        #%%%%%%% Sine
        H = sin(tempH)
    elif __switch_0__ == mcellarray([mstring('hardlim')]):
        #%%%%%%% Hard Limit
        H = hardlim(tempH)
        #%%%%%%% More activation functions can be added here                
    end
    clear(mstring('tempH'))#   Release the temparary array for calculation of hidden neuron output matrix H

    #%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
    OutputWeight = pinv(H.cT) * T.cT
    end_time_train = cputime
    TrainingTime = end_time_train - start_time_train; 
    print(TrainingTime)#   Calculate CPU time (seconds) spent for training ELM

    #%%%%%%%%%% Calculate the training accuracy
    Y = (H.cT * OutputWeight).cT#   Y: the actual output of the training data
    if Elm_Type == REGRESSION:
        TrainingAccuracy = sqrt(mse(T - Y)); 
        print(TrainingAccuracy)    #   Calculate training accuracy (RMSE) for regression case
        output = Y
    end
    clear(mstring('H'))

    if Elm_Type == CLASSIFIER:
        #%%%%%%%%% Calculate training & testing classification accuracy
        MissClassificationRate_Training = 0

        for i in mslice[1:size(T, 2)]:
            [x, label_index_expected] = max(T(mslice[:], i))
            [x, label_index_actual] = max(Y(mslice[:], i))
            output(i).lvalue = label(label_index_actual)
            if label_index_actual != label_index_expected:
                MissClassificationRate_Training = MissClassificationRate_Training + 1
            end
        end
        TrainingAccuracy = 1 - MissClassificationRate_Training / NumberofTrainingData; 
        print(TrainingAccuracy)
    end

    if Elm_Type != REGRESSION:
        save(mstring('elm_model'), mstring('NumberofInputNeurons'), mstring('NumberofOutputNeurons'), mstring('InputWeight'), mstring('BiasofHiddenNeurons'), mstring('OutputWeight'), mstring('ActivationFunction'), mstring('label'), mstring('Elm_Type'))
    else:
        save(mstring('elm_model'), mstring('InputWeight'), mstring('BiasofHiddenNeurons'), mstring('OutputWeight'), mstring('ActivationFunction'), mstring('Elm_Type'))
    end