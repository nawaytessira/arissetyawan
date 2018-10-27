(mstring('<include>ELM.m</include>'))

test = zeros(50, 1)
train = zeros(50, 1)
train_time = zeros(50, 1)
testing_time = zeros(50, 1)

wb = waitbar(0, mstring('Please waiting...'))
# default to 50 iter trial
for rnd in mslice[1:3]:

    waitbar(rnd / 50, wb)

    # diabetes2_data;     %   randomly generate new training and testing data for every trial of simulation
    [learn_time, test_time, train_accuracy, test_accuracy] = ELM(mstring('diabetes_train'), mstring('diabetes_test'), 1, 20, mstring('sig'))
    test(rnd, 1).lvalue = test_accuracy
    train(rnd, 1).lvalue = train_accuracy
    train_time(rnd, 1).lvalue = learn_time
    testing_time(rnd, 1).lvalue = test_time
end
close(wb)

AverageTrainingTime = mean(train_time); print AverageTrainingTime
StandardDeviationofTrainingTime = std(train_time); print StandardDeviationofTrainingTime
AvergeTestingTime = mean(testing_time); print AvergeTestingTime
StandardDeviationofTestingTime = std(testing_time); print StandardDeviationofTestingTime
AverageTrainingAccuracy = mean(train); print AverageTrainingAccuracy
StandardDeviationofTrainingAccuracy = std(train); print StandardDeviationofTrainingAccuracy
AverageTestingAccuracy = mean(test); print AverageTestingAccuracy
StandardDeviationofTestingAccuracy = std(test); print StandardDeviationofTestingAccuracy