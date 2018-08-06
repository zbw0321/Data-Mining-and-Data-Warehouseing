
import helper
import math
import numpy as np
from sklearn import svm


def fool_classifier(test_data):  ## Please do not change the function defination...

    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance = helper.strategy()
    parameters = {'gamma': 'auto', 'C': 4, 'kernel': 'linear', 'degree': 3, 'coef0': 0.0}

    def getTrainingVocalDict(allTrainingText):
        vocalDict = {}
        count = 0
        for eachList in allTrainingText:
            for word in eachList:
                count = count + 1
                if word not in vocalDict:
                    vocalDict[word] = 1
                else:
                    vocalDict[word] = vocalDict[word] + 1
        vocalList = list(vocalDict.keys())
        return vocalDict, vocalList, count

    ######################## 1. Count vocabulary  #########################
    #  count:     vocabulary Size
    #  vocalList: vocabulary list
    #  vocalDict: vocabulary dict. key=word, value= word freq in all train_data

    allTrainingText = strategy_instance.class0 + strategy_instance.class1
    vocalDict, vocalList, count = getTrainingVocalDict(allTrainingText)

    ######################## 2. Process features  ########################
    # Use Tfiwf to transform words into number features
    # Tf = one single word frequency in on sample / len(sample)
    # Iwf = log (all words in training data / the number of one single word shows up in all training data)
    # Tfiwf = Tf * Iwf

    def countWordFreq(Text, vocalList):
        wordFreqVector = [[0 for j in range(len(vocalList))] for i in range(len(Text))]
        for w in range(len(vocalList)):
            for i in range(len(Text)):
                for j in range(len(Text[i])):
                    if vocalList[w] == Text[i][j]:
                        wordFreqVector[i][w] = wordFreqVector[i][w] + 1
        vecter = []
        for i in range(len(Text)):
            tmp = []
            eachLen = len(Text[i])
            for each in wordFreqVector[i]:
                each = each / eachLen
                tmp.append(each)
            vecter.append(tmp)
        return vecter

    def getIWF(vocalDict, vocalList, Text, count):
        iwf_vector = []
        for i in range(len(Text)):
            tmp = []
            for word in vocalList:
                eachIwf = math.log(count / vocalDict[word])
                tmp.append(eachIwf)
            iwf_vector.append(tmp)
        return iwf_vector

    def getTfiwf(oneClassTF_vector, oneClassIwf_vector):
        tfiwfForEachClass = []
        for i in range(len(oneClassTF_vector)):
            eachDoc = []
            for j in range(len(oneClassTF_vector[i])):
                eachTfiwf = oneClassTF_vector[i][j] * oneClassIwf_vector[i][j]
                eachDoc.append(eachTfiwf)
            tfiwfForEachClass.append(eachDoc)
        return tfiwfForEachClass

    class0Text = strategy_instance.class0
    class1Text = strategy_instance.class1
    oneClassTF_vector0 = countWordFreq(class0Text, vocalList)
    oneClassTF_vector1 = countWordFreq(class1Text, vocalList)
    oneClassIwf_vector0 = getIWF(vocalDict, vocalList, class0Text, count)
    oneClassIwf_vector1 = getIWF(vocalDict, vocalList, class1Text, count)

    x_train0 = getTfiwf(oneClassTF_vector0, oneClassIwf_vector0)
    x_train1 = getTfiwf(oneClassTF_vector1, oneClassIwf_vector1)
    x_train = x_train0 + x_train1

    y_train0 = [0 for i in range(len(class0Text))]
    y_train1 = [1 for i in range(len(class1Text))]
    y_train = y_train0 + y_train1

    ############################# 3. AdaBoost train：  #################################
    # conjunction with 16 svm linear classifiers
    # 11 of them process 1:1 train data from class0 and class1. Each time these classifiers get random train data from
    # 2 of them process half train data based on the txt file sequences
    # 3 of them process the whole train data set
    # All these classifiers work as a super classifier to boost the performance.
    # coef of the super classifier is the mean of 16 coefs of each weak classifiers

    def getDataByRandomIndex(random_indices0, random_indices1, x_train0, x_train1, y_train0, y_train1):
        x_train_random_all = []
        y_train_random_all = []
        for i in range(len(random_indices0)):
            x_tmp0 = []
            x_tmp1 = []
            y_tmp0 = []
            y_tmp1 = []

            for j in range(len(random_indices0[i])):
                x_tmp0.append(x_train0[random_indices0[i][j]])
                x_tmp1.append(x_train1[random_indices1[i][j]])
                y_tmp0.append(y_train0[random_indices0[i][j]])
                y_tmp1.append(y_train1[random_indices1[i][j]])
            x_train_random_all.append(x_tmp0 + x_tmp1)
            y_train_random_all.append(y_tmp0 + y_tmp1)

        return x_train_random_all, y_train_random_all

    minSample = min(len(x_train0), len(x_train1))

    random_indices_class0 = np.random.randint(0, len(x_train0), [11, minSample]).tolist()
    random_indices_class1 = np.random.randint(0, len(x_train1), [11, minSample]).tolist()
    x_train_random_all, y_train_random_all = getDataByRandomIndex(random_indices_class0, random_indices_class1,
                                                                  x_train0, x_train1, y_train0, y_train1)

    clf0 = strategy_instance.train_svm(parameters, np.asarray(x_train_random_all[0]), np.asarray(y_train_random_all[0]))
    clf1 = strategy_instance.train_svm(parameters, np.asarray(x_train_random_all[1]), np.asarray(y_train_random_all[1]))
    clf2 = strategy_instance.train_svm(parameters, np.asarray(x_train_random_all[2]), np.asarray(y_train_random_all[2]))
    clf3 = strategy_instance.train_svm(parameters, np.asarray(x_train_random_all[3]), np.asarray(y_train_random_all[3]))
    clf4 = strategy_instance.train_svm(parameters, np.asarray(x_train_random_all[4]), np.asarray(y_train_random_all[4]))
    clf5 = strategy_instance.train_svm(parameters, np.asarray(x_train_random_all[5]), np.asarray(y_train_random_all[5]))
    clf6 = strategy_instance.train_svm(parameters, np.asarray(x_train_random_all[6]), np.asarray(y_train_random_all[6]))
    clf7 = strategy_instance.train_svm(parameters, np.asarray(x_train_random_all[7]), np.asarray(y_train_random_all[7]))
    clf8 = strategy_instance.train_svm(parameters, np.asarray(x_train_random_all[8]), np.asarray(y_train_random_all[8]))
    clf9 = strategy_instance.train_svm(parameters, np.asarray(x_train_random_all[9]), np.asarray(y_train_random_all[9]))
    clf10 = strategy_instance.train_svm(parameters, np.asarray(x_train_random_all[10]),
                                        np.asarray(y_train_random_all[10]))

    clf11 = strategy_instance.train_svm(parameters, np.asarray(x_train0[0:minSample] + x_train1),
                                        np.asarray(y_train0[0:minSample] + y_train1))
    clf12 = strategy_instance.train_svm(parameters,
                                        np.asarray(x_train0[len(x_train0) - minSample:len(x_train0)] + x_train1),
                                        np.asarray(y_train0[len(y_train0) - minSample:len(y_train0)] + y_train1))
    clf13 = strategy_instance.train_svm(parameters, np.asarray(x_train0 + x_train1), np.asarray(y_train0 + y_train1))
    clf14 = strategy_instance.train_svm(parameters, np.asarray(x_train0 + x_train1), np.asarray(y_train0 + y_train1))
    clf15 = strategy_instance.train_svm(parameters, np.asarray(x_train0 + x_train1), np.asarray(y_train0 + y_train1))

    x_train_random_all.append(x_train0[0:minSample] + x_train1)
    y_train_random_all.append(y_train0[0:minSample] + y_train1)

    x_train_random_all.append(x_train0[len(x_train0) - minSample:len(x_train0)] + x_train1)
    y_train_random_all.append(y_train0[len(x_train0) - minSample:len(x_train0)] + y_train1)

    x_train_random_all.append(x_train0 + x_train1)
    y_train_random_all.append(y_train0 + y_train1)
    x_train_random_all.append(x_train0 + x_train1)
    y_train_random_all.append(y_train0 + y_train1)
    x_train_random_all.append(x_train0 + x_train1)
    y_train_random_all.append(y_train0 + y_train1)

    ###########  cross_validation ##############
    # clf_init = svm.SVC(kernel="linear", C=parameters['C'])
    # from sklearn.cross_validation import cross_val_score
    # sum_CV = 0
    # for i in range(len(x_train_random_all)):
    #     scores_cv = cross_val_score(clf_init, np.asarray(x_train_random_all[i]), np.asarray(y_train_random_all[i]),
    #                                 cv=5, scoring="accuracy")
    #     print("scores_cv", i, ".mean=", scores_cv.mean())
    #     sum_CV = sum_CV + scores_cv.mean()
    # mean_CV = sum_CV / len(x_train_random_all)
    # print("mean_CV  =", mean_CV)
    ########### cross_validation##############

    coefs0 = clf0.coef_
    coefs1 = clf1.coef_
    coefs2 = clf2.coef_
    coefs3 = clf3.coef_
    coefs4 = clf4.coef_
    coefs5 = clf5.coef_
    coefs6 = clf6.coef_
    coefs7 = clf7.coef_
    coefs8 = clf8.coef_
    coefs9 = clf9.coef_
    coefs10 = clf10.coef_

    coefs11 = clf11.coef_
    coefs12 = clf12.coef_
    coefs13 = clf13.coef_
    coefs14 = clf14.coef_
    coefs15 = clf15.coef_
    coefs = (coefs0 + coefs1 + coefs2 + coefs3 + coefs4 + coefs5 + coefs6 + coefs7 + coefs8 + coefs9 + coefs10 + coefs11 + coefs12 + coefs13 + coefs14 + coefs15) / 16

    ############################# 4. AdaBoost test: ##################################
    # Transform Tfiwf to test data set and turn words into numbers (features)
    # predict the test data set


    with open(test_data, 'r') as test_data2:
        test_data_list = [line.strip().split(' ') for line in test_data2]

    def getTransformTestVocalCount(vocalList, testText):
        count_test = 0
        for eachList in testText:
            for eachWord in eachList:
                if eachWord in vocalList:
                    count_test = count_test + 1
        return count_test

    class1Text_test = test_data_list
    count_test = getTransformTestVocalCount(vocalList, class1Text_test)
    oneClassTF_vector1_test = countWordFreq(class1Text_test, vocalList)
    oneClassIwf_vector1_test = getIWF(vocalDict, vocalList, class1Text_test, count_test)
    x_test = getTfiwf(oneClassTF_vector1_test, oneClassIwf_vector1_test)

    x_test = np.asarray(x_test)
    y_target = np.array([1 for i in range(len(class1Text_test))])

    y_pred0 = clf0.predict(x_test)
    y_pred1 = clf1.predict(x_test)
    y_pred2 = clf2.predict(x_test)
    y_pred3 = clf3.predict(x_test)
    y_pred4 = clf4.predict(x_test)
    y_pred5 = clf5.predict(x_test)
    y_pred6 = clf6.predict(x_test)
    y_pred7 = clf7.predict(x_test)
    y_pred8 = clf8.predict(x_test)
    y_pred9 = clf9.predict(x_test)
    y_pred10 = clf10.predict(x_test)

    y_pred11 = clf11.predict(x_test)
    y_pred12 = clf12.predict(x_test)
    y_pred13 = clf13.predict(x_test)
    y_pred14 = clf14.predict(x_test)
    y_pred15 = clf15.predict(x_test)

    y_pred_list = [y_pred0, y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, y_pred7, y_pred8, y_pred9, y_pred10,
                   y_pred11, y_pred12, y_pred13, y_pred14, y_pred15]

    # from sklearn.metrics import confusion_matrix
    # for pred in y_pred_list:
    #     cm = confusion_matrix(y_target, pred)
    #     print("cm = ", cm)

    from sklearn.metrics import mean_squared_error
    from sklearn import metrics
    accuracy_list = []
    loss_list = []
    for pred in y_pred_list:
        loss = mean_squared_error(y_target, pred)
        # print("pred loss=", loss)
        accuracy = metrics.accuracy_score(y_target, pred)
        # print("pred accuracy=", accuracy)
        # print()
        accuracy_list.append(accuracy)
        loss_list.append(loss)

    # print("16 pred.mean.accuracy = ", np.mean(accuracy_list))
    # print("16 pred.mean.loss     = ", np.mean(loss_list))

    ################################ 5. voting ############################################
    # Use voting for get the final predict of the super classifer
    voting_pred = []
    for i in range(len(y_pred0)):
        zero_num = 0
        one_num = 0
        for j in range(len(y_pred_list)):
            if y_pred_list[j][i] == 0:
                zero_num = zero_num + 1
            else:
                one_num = one_num + 1
        if zero_num > one_num:
            voting = 0
        else:
            voting = 1
        voting_pred.append(voting)

    voting_accuracy = metrics.accuracy_score(y_target, voting_pred)
    # print("voting_accuracy=", voting_accuracy)
    # print("C=", parameters["C"])
    ######################## 6. show top10 and down10 coefs ###############
    # sort
    coefs_sort_indices = coefs.argsort()
    top10 = coefs_sort_indices[0][len(vocalList) - 10:len(vocalList)].tolist()
    down10 = coefs_sort_indices[0][0:10].tolist()
    item20 = down10[::-1] + top10[::-1]
    result = []

    for index in item20:
        result.append((coefs[0][index], vocalList[index]))

    for item in result:
        print(item)

    ######################## 7. modify test_data.txt ###############
    # for each sample in test_data.txt, find 10 words perform as the most important features in class1.
    # use the most top10 important features in class0 (which are not appear in the sample test data) to replace


    down5718 = coefs_sort_indices[0].tolist()

    coefs_sort_num = coefs_sort_indices.tolist()
    vocal_sort = []
    ###  find 10 words need to change
    for index in coefs_sort_num[0]:
        vocal_sort.append(vocalList[index])

    vocal_sort_high_to_low = vocal_sort[::-1]
    need_change_words = []
    for i in range(len(test_data_list)):
        tmp = []
        for eachvocal in vocal_sort_high_to_low:
            for word in set(test_data_list[i]):
                if len(tmp) == 10:
                    break
                if word == eachvocal and word not in tmp:
                    tmp.append(word)
        need_change_words.append(tmp)

    test_moddify_data_list = test_data_list[::1]

    def getOneNewWord(down5718, doc, i, j, k, vocalList):
        for index in range(len(down5718)):
            if vocalList[down5718[index]] not in doc:
                # print("volcalList[down5718[index]] =",volcalList[down5718[index]])
                return vocalList[down5718[index]]

    for i in range(len(test_moddify_data_list)):  # 200
        for k in range(len(need_change_words[i])):  # 10
            tmpDict = {}
            for j in range(len(test_moddify_data_list[i])):
                if need_change_words[i][k] == test_moddify_data_list[i][j]:
                    if need_change_words[i][k] in tmpDict:
                        newWord = tmpDict[need_change_words[i][k]]
                    else:
                        newWord = getOneNewWord(down5718, test_moddify_data_list[i], i, j, k, vocalList)
                        tmpDict[need_change_words[i][k]] = newWord
                    test_moddify_data_list[i][j] = newWord

    output = test_moddify_data_list

    ################### 8. write file  ########################
    modified_data = './modified_data.txt'
    filename = modified_data
    with open(filename, 'w') as f:
        for i in range(len(output)):
            text = ""
            for j in range(len(output[i])):
                word = output[i][j]
                text = text + word + " "

            f.write(text + "\n")

    ################ 9. check data #####################

    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    ## You can check that the modified text is within the modification limits.
    modified_data = './modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance  ## NOTE: You are required to return the instance of this class.

