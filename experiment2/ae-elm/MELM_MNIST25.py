from loadMNISTImages import *
from loadMNISTLabels import *


# @mfunction("TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy")
def MELM_MNIST25(Testcode=None, TrainDataSize=None, TotalLayers=None, HiddernNeurons=None, C1=None, rhoValue=None, sigpara=None, sigpara1=None):

    # Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
    # OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
    #
    # Input:
    # TrainingData_File     - Filename of training data set
    # TestingData_File      - Filename of testing data set
    # Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
    # NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
    # ActivationFunction    - Type of activation function:
    #                           'sig' for Sigmoidal function
    #                           'sin' for Sine function
    #                           'hardlim' for Hardlim function
    #                           'tribas' for Triangular basis function
    #                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
    #
    # Output: 
    # TrainingTime          - Time (seconds) spent on training ELM
    # TestingTime           - Time (seconds) spent on predicting ALL testing data
    # TrainingAccuracy      - Training accuracy: 
    #                           RMSE for regression or correct classification rate for classification
    # TestingAccuracy       - Testing accuracy: 
    #                           RMSE for regression or correct classification rate for classification
    #
    # MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
    # FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
    # neurons; neuron 5 has the highest output means input belongs to 5-th class
    #
    # Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm('sinc_train', 'sinc_test', 0, 20, 'sig')
    # Sample2 classification: elm('diabetes_train', 'diabetes_test', 1, 20, 'sig')
    #
    #%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    #%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    #%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    #%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    #%%%    DATE:       APRIL 2004



    #%%%%%%%%%% Load training dataset
    P = loadMNISTImages(mstring('mnist/train-images.idx3-ubyte'))
    T = loadMNISTLabels(mstring('mnist/train-labels.idx1-ubyte'))
    T = T.cT


    if TrainDataSize != 0:
        rand_sequence = randperm(TrainDataSize)
        temp_P = P.cT
        temp_T = T.cT
        clear(mstring('P'))
        clear(mstring('T'))
        P = temp_P(rand_sequence, mslice[:]).cT
        T = temp_T(rand_sequence, mslice[:]).cT
        clear(mstring('temp_P'))
        clear(mstring('temp_T'))
    end

    #%%%%%%%%%% Load testing dataset
    TV.T = loadMNISTLabels(mstring('mnist/t10k-labels.idx1-ubyte'))
    TV.T = TV.T.cT#   Release raw testing data array



    NumberofTrainingData = size(P, 2)
    NumberofTestingData = size(TV.T, 2)
    NumberofInputNeurons = size(P, 1)



    #%%%%%%%%%%% Preprocessing the data of classification
    sorted_target = sort(cat(2, T, TV.T), 2)
    label = zeros(1, 1)#   Find and save in 'label' class label from training and testing data sets
    label(1, 1).lvalue = sorted_target(1, 1)
    j = 1
    for i in mslice[2:(NumberofTrainingData + NumberofTestingData)]:
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

    #%%%%%%%%% Processing the targets of testing
    temp_TV_T = zeros(NumberofOutputNeurons, NumberofTestingData)
    for i in mslice[1:NumberofTestingData]:
        for j in mslice[1:number_class]:
            if label(1, j) == TV.T(1, i):
                break
            end
        end
        temp_TV_T(j, i).lvalue = 1
    end
    TV.T = temp_TV_T * 2 - 1



    #%%%%%%%%%% Calculate weights & biases
    train_time = tic
    no_Layers = TotalLayers
    stack = cell(no_Layers + 1, 1)

    lenHN = length(HiddernNeurons)
    lenC1 = length(C1)
    lensig = length(sigpara)
    lensig1 = length(sigpara1)
    HN_temp = mcat([NumberofInputNeurons, HiddernNeurons(mslice[1:lenHN - 1])])
    if length(HN_temp) < no_Layers:
        HN = mcat([HN_temp, repmat(HN_temp(length(HN_temp)), 1, no_Layers - length(HN_temp)), HiddernNeurons(lenHN)])
        C = mcat([C1(mslice[1:lenC1 - 2]), zeros(1, no_Layers - length(HN_temp)), C1(mslice[lenC1 - 1:lenC1])])
        #         sigscale = [sigpara(1:lensig-1),repmat(sigpara(lensig-1),1,no_Layers - length(HN_temp)),sigpara(lensig)];
        sigscale = mcat([sigpara(mslice[1:lensig - 1]), ones(1, no_Layers - length(HN_temp)), sigpara(lensig)])
        sigscale1 = mcat([sigpara1(mslice[1:lensig1 - 1]), ones(1, no_Layers - length(HN_temp)), sigpara1(lensig1)])
    else:
        HN = mcat([NumberofInputNeurons, HiddernNeurons])
        C = C1
        sigscale = sigpara
        sigscale1 = sigpara1
    end
    clear(mstring('HN_temp'))

    InputDataLayer = zscore(P)
    #     InputDataLayer = P;
    clear(mstring('P'))

    #     s=rng;
    if Testcode == 1:
        rng(mstring('default'))
        #         randomWeightRng = 0;
        #     else
        #         randomWeightRng = s.Seed;
    end
    #     prevRowWeight = -1;
    #     prevColWeight = -1;
    for i in mslice[1:1:no_Layers]:

        #         if prevRowWeight ~= HN(i+1) || prevColWeight ~= HN(i)
        #             randomWeightRng = randomWeightRng +1;
        #         end
        #         prevRowWeight= HN(i+1);
        #         prevColWeight = HN(i);
        #         rng(randomWeightRng);
        InputWeight = rand(HN(i + 1), HN(i)) * 2 - 1
        if HN(i + 1) > HN(i):
            InputWeight = orth(InputWeight)
        else:
            InputWeight = orth(InputWeight.cT).cT
        end
        #         rng(randomWeightRng);
        BiasofHiddenNeurons = rand(HN(i + 1), 1) * 2 - 1
        BiasofHiddenNeurons = orth(BiasofHiddenNeurons)

        tempH = InputWeight * InputDataLayer    #   Release input of training data
        clear(mstring('InputWeight'))
        ind = ones(1, NumberofTrainingData)
        BiasMatrix = BiasofHiddenNeurons(mslice[:], ind)    #   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
        tempH = tempH + BiasMatrix

        clear(mstring('BiasMatrix'))
        clear(mstring('BiasofHiddenNeurons'))
        fprintf(1, mstring('AutoEncorder Max Val %f Min Val %f\\n'), max(tempH(mslice[:])), min(tempH(mslice[:])))
        H = 1 /eldiv/ (1 + exp(-sigscale1(i) * tempH))
        clear(mstring('tempH'))    #   Release the temparary array for calculation of hidden neuron output matrix H

        #%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
        if HN(i + 1) == HN(i):
            stack(i).w            
            procrustNew(InputDataLayer.cT, H.cT)
        else:
            if C(i) == 0:
                stack(i).w.lvalue = pinv(H.cT) * InputDataLayer.cT            # implementation without regularization factor //refer to 2006 Neurocomputing paper
            else:
                rhohats = mean(H, 2)
                rho = rhoValue
                KLsum = sum(rho * log(rho /eldiv/ rhohats) + (1 - rho) * log((1 - rho) /eldiv/ (1 - rhohats)))

                Hsquare = H * H.cT
                HsquareL = diag(max(Hsquare, mcat([]), 2))
                stack(i).w.lvalue = ((eye(size(H, 1)) *elmul* KLsum + HsquareL) * (1 / C(i)) + Hsquare)
                print(stack)
                (H * InputDataLayer.cT)


                clear("Hsquare")
                clear("HsquareL")
            end
        end

        tempH = (stack(i).w) * (InputDataLayer)
        clear("InputDataLayer")

        if HN(i + 1) == HN(i):
            InputDataLayer = tempH
        else:
            fprintf(1, mstring('Layered Max Val %f Min Val %f\\n'), max(tempH(mslice[:])), min(tempH(mslice[:])))
            InputDataLayer = 1 /eldiv/ (1 + exp(-sigscale(i) * tempH))
        end

        clear("tempH")
        clear("H")
    end


    if C(no_Layers + 1) == 0:
        stack(no_Layers + 1).w.lvalue = pinv(InputDataLayer.cT) * T.cT
    else:
        stack(no_Layers + 1).w.lvalue = (eye(size(InputDataLayer, 1)) / C(no_Layers + 1) + InputDataLayer * InputDataLayer.cT);
        print(stack)
        (InputDataLayer * T.cT)

    end



    TrainingTime = toc(train_time)#   Calculate CPU time (seconds) spent for training ELM
    #     display_network(orth(stack{1}.w'));
    #%%%%%%%%%% Calculate the training accuracy
    Y = (InputDataLayer.cT * stack(no_Layers + 1).w).cT#   Y: the actual output of the training data

    clear("InputDataLayer")
    clear("H")

    TV.P = loadMNISTImages(mstring('mnist/t10k-images.idx3-ubyte'))
    #%%%%%%%%%% Calculate the output of testing input
    test_time = tic
    InputDataLayer = zscore(TV.P)
    #     InputDataLayer = TV.P;
    P

    for i in mslice[1:1:no_Layers]:
        tempH_test = (stack(i).w) * (InputDataLayer)
        P    #   Release input of testing data

        if HN(i + 1) == HN(i):
            InputDataLayer = tempH_test
        else:
            InputDataLayer = 1 /eldiv/ (1 + exp(-sigscale(i) * tempH_test))
        end
        clear("tempH_test")

    end

    TY = (InputDataLayer.cT * stack(no_Layers + 1).w).cT#   TY: the actual output of the testing data

    TestingTime = toc(test_time)#   Calculate CPU time (seconds) spent by ELM predicting the whole testing data



    #%%%%%%%%% Calculate training & testing classification accuracy
    MissClassificationRate_Training = 0
    MissClassificationRate_Testing = 0

    for i in mslice[1:size(T, 2)]:
        max(T(mslice[:], i))
        [x, label_index_actual] = max(Y(mslice[:], i))
        if label_index_actual != label_index_expected:
            MissClassificationRate_Training = MissClassificationRate_Training + 1
        end
    end
    TrainingAccuracy = 1 - MissClassificationRate_Training / size(T, 2)
    for i in mslice[1:size(TV.T, 2)]:
        [x, label_index_expected] = max(TV.T(mslice[:], i))
        [x, label_index_actual] = max(TY(mslice[:], i))
        if label_index_actual != label_index_expected:
            MissClassificationRate_Testing = MissClassificationRate_Testing + 1
        end
    end
    TestingAccuracy = 1 - MissClassificationRate_Testing / size(TV.T, 2)


# end