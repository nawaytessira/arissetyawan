@mfunction("TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy, TY")
def elm_kernel(TrainingData_File=None, TestingData_File=None, Elm_Type=None, Regularization_coefficient=None, Kernel_type=None, Kernel_para=None):

    # Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
    # OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
    #
    # Input:
    # TrainingData_File           - Filename of training data set
    # TestingData_File            - Filename of testing data set
    # Elm_Type                    - 0 for regression; 1 for (both binary and multi-classes) classification
    # Regularization_coefficient  - Regularization coefficient C
    # Kernel_type                 - Type of Kernels:
    #                                   'RBF_kernel' for RBF Kernel
    #                                   'lin_kernel' for Linear Kernel
    #                                   'poly_kernel' for Polynomial Kernel
    #                                   'wav_kernel' for Wavelet Kernel
    #Kernel_para                  - A number or vector of Kernel Parameters. eg. 1, [0.1,10]...
    # Output: 
    # TrainingTime                - Time (seconds) spent on training ELM
    # TestingTime                 - Time (seconds) spent on predicting ALL testing data
    # TrainingAccuracy            - Training accuracy: 
    #                               RMSE for regression or correct classification rate for classification
    # TestingAccuracy             - Testing accuracy: 
    #                               RMSE for regression or correct classification rate for classification
    #
    # MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
    # FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
    # neurons; neuron 5 has the highest output means input belongs to 5-th class
    #
    # Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm_kernel('sinc_train', 'sinc_test', 0, 1, ''RBF_kernel',100)
    # Sample2 classification: elm_kernel('diabetes_train', 'diabetes_test', 1, 1, 'RBF_kernel',100)
    #
    #%%%    Authors:    MR HONG-MING ZHOU AND DR GUANG-BIN HUANG
    #%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    #%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    #%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    #%%%    DATE:       MARCH 2012

    #%%%%%%%%%% Macro definition
    REGRESSION = 0
    CLASSIFIER = 1

    #%%%%%%%%%% Load training dataset
    train_data = load(TrainingData_File)
    T = train_data(mslice[:], 1).cT
    P = train_data(mslice[:], mslice[2:size(train_data, 2)]).cT
    clear(mstring('train_data'))#   Release raw training data array

    #%%%%%%%%%% Load testing dataset
    test_data = load(TestingData_File)
    TV.T = test_data(mslice[:], 1).cT
    TV.P = test_data(mslice[:], mslice[2:size(test_data, 2)]).cT
    clear(mstring('test_data'))#   Release raw testing data array

    C = Regularization_coefficient
    NumberofTrainingData = size(P, 2)
    NumberofTestingData = size(TV.P, 2)

    if Elm_Type != REGRESSION:
        #%%%%%%%%%%% Preprocessing the data of classification
        sorted_target = sort(cat(2, T, TV.T), 2)
        label = zeros(1, 1)    #   Find and save in 'label' class label from training and testing data sets
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
        #   end if of Elm_Type
    end

    #%%%%%%%%%% Training Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic()
    n = size(T, 2)
    Omega_train = kernel_matrix(P.cT, Kernel_type, Kernel_para)

    TrainingTime = toc; print TrainingTime

    #%%%%%%%%%% Calculate the training output
    Y = (Omega_train * OutputWeight).cT#   Y: the actual output of the training data

    #%%%%%%%%%% Calculate the output of testing input
    tic()
    Omega_test = kernel_matrix(P.cT, Kernel_type, Kernel_para, TV.P.cT)
    TY = (Omega_test.cT * OutputWeight).cT#   TY: the actual output of the testing data
    TestingTime = toc; print TestingTime

    #%%%%%%%%% Calculate training & testing classification accuracy

    if Elm_Type == REGRESSION:
        #%%%%%%%%% Calculate training & testing accuracy (RMSE) for regression case
        TrainingAccuracy = sqrt(mse(T - Y)); print TrainingAccuracy
        TestingAccuracy = sqrt(mse(TV.T - TY)); print TestingAccuracy
    end

    if Elm_Type == CLASSIFIER:
        #%%%%%%%%% Calculate training & testing classification accuracy
        MissClassificationRate_Training = 0
        MissClassificationRate_Testing = 0

        for i in mslice[1:size(T, 2)]:
            [x, label_index_expected] = max(T(mslice[:], i))
            [x, label_index_actual] = max(Y(mslice[:], i))
            if label_index_actual != label_index_expected:
                MissClassificationRate_Training = MissClassificationRate_Training + 1
            end
        end
        TrainingAccuracy = 1 - MissClassificationRate_Training / size(T, 2); print TrainingAccuracy
        for i in mslice[1:size(TV.T, 2)]:
            [x, label_index_expected] = max(TV.T(mslice[:], i))
            [x, label_index_actual] = max(TY(mslice[:], i))
            if label_index_actual != label_index_expected:
                MissClassificationRate_Testing = MissClassificationRate_Testing + 1
            end
        end
        TestingAccuracy = 1 - MissClassificationRate_Testing / size(TV.T, 2); print TestingAccuracy
    end


    #%%%%%%%%%%%%%%%%% Kernel Matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@mfunction("omega")
def kernel_matrix(Xtrain=None, kernel_type=None, kernel_pars=None, Xt=None):

    nb_data = size(Xtrain, 1)


    if strcmp(kernel_type, mstring('RBF_kernel')):
        if nargin < 4:
            XXh = sum(Xtrain **elpow** 2, 2) * ones(1, nb_data)
            omega = XXh + XXh.cT - 2 * (Xtrain * Xtrain.cT)
            omega = exp(-omega /eldiv/ kernel_pars(1))
        else:
            XXh1 = sum(Xtrain **elpow** 2, 2) * ones(1, size(Xt, 1))
            XXh2 = sum(Xt **elpow** 2, 2) * ones(1, nb_data)
            omega = XXh1 + XXh2.cT - 2 * Xtrain * Xt.cT
            omega = exp(-omega /eldiv/ kernel_pars(1))
        end

    elif strcmp(kernel_type, mstring('lin_kernel')):
        if nargin < 4:
            omega = Xtrain * Xtrain.cT
        else:
            omega = Xtrain * Xt.cT
        end

    elif strcmp(kernel_type, mstring('poly_kernel')):
        if nargin < 4:
            omega = (Xtrain * Xtrain.cT + kernel_pars(1)) **elpow** kernel_pars(2)
        else:
            omega = (Xtrain * Xt.cT + kernel_pars(1)) **elpow** kernel_pars(2)
        end

    elif strcmp(kernel_type, mstring('wav_kernel')):
        if nargin < 4:
            XXh = sum(Xtrain **elpow** 2, 2) * ones(1, nb_data)
            omega = XXh + XXh.cT - 2 * (Xtrain * Xtrain.cT)

            XXh1 = sum(Xtrain, 2) * ones(1, nb_data)
            omega1 = XXh1 - XXh1.cT
            omega = cos(kernel_pars(3) * omega1 /eldiv/ kernel_pars(2)) *elmul* exp(-omega /eldiv/ kernel_pars(1))

        else:
            XXh1 = sum(Xtrain **elpow** 2, 2) * ones(1, size(Xt, 1))
            XXh2 = sum(Xt **elpow** 2, 2) * ones(1, nb_data)
            omega = XXh1 + XXh2.cT - 2 * (Xtrain * Xt.cT)

            XXh11 = sum(Xtrain, 2) * ones(1, size(Xt, 1))
            XXh22 = sum(Xt, 2) * ones(1, nb_data)
            omega1 = XXh11 - XXh22.cT

            omega = cos(kernel_pars(3) * omega1 /eldiv/ kernel_pars(2)) *elmul* exp(-omega /eldiv/ kernel_pars(1))
        end
    end