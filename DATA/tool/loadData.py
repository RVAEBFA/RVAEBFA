import numpy as np

def loadData(file_name, ratio=0.8):
    data = np.load(file_name)
    labels = data["Y"]
    labels = labels.flatten()  #折叠成一维数组
    features = data["X"]
    normal_data = features[labels==-1]   #inliers
    normal_labels = labels[labels==-1]
    # normal_labels=normal_labels* (-1)

    abnormal_data = features[labels==1]  #outliers
    abnormal_labels = labels[labels==1]
    # abnormal_labels=abnormal_labels * (-1)
    # print('normal size:', abnormal_data.shape[0])
    # print('abnormal size:', normal_data.shape[0])
    
    normal_length = normal_data.shape[0]
    abnormal_length = abnormal_data.shape[0]
    print("normal_length",normal_length)
    print("abnormal_length",abnormal_length)
    normal_end_index = int(normal_length * ratio)
    abnormal_end_index = int(abnormal_length * ratio)
    print(normal_end_index)
    print(abnormal_end_index)
    # trainData = np.r_[normal_data[:normal_end_index, :], abnormal_data[:abnormal_end_index, :]]
    # trainLabels = np.r_[normal_labels[:normal_end_index], abnormal_labels[:abnormal_end_index]]
    trainData = np.r_[normal_data[:abnormal_end_index, :], abnormal_data[:abnormal_end_index, :]]
    trainLabels = np.r_[normal_labels[:abnormal_end_index], abnormal_labels[:abnormal_end_index]]

    testData = np.r_[normal_data[(normal_data.shape[0] - int(abnormal_data.shape[0] * 0.2)) :, :], abnormal_data[abnormal_end_index:, :]]
    testLabels = np.r_[normal_labels[(normal_data.shape[0] - int(abnormal_data.shape[0] * 0.2)) :], abnormal_labels[abnormal_end_index:]]

    print(trainData.shape)
    print(testData.shape)

    # testData = np.r_[normal_data[normal_end_index:, :], abnormal_data]
    # testLabels = np.r_[normal_labels[normal_end_index:], abnormal_labels]

    return trainData, trainLabels, testData, testLabels
