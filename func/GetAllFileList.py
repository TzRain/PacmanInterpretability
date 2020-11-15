import pickle

if __name__ == '__main__':
    trainSet = set()
    pkl_file = open("../result/trainList.pkl", 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    dataList = data["dataList"]
    for item in dataList:
        path = item["path"]
        name = path.rsplit("/", 1)[1]
        nameSplit = name.split("-")
        trainSet.add("-".join(nameSplit[0: -1]))

    pkl_file = open("../result/testList.pkl", 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    dataList = data["dataList"]
    for item in dataList:
        path = item["path"]
        name = path.rsplit("/", 1)[1]
        nameSplit = name.split("-")
        trainSet.add("-".join(nameSplit[0: -1]))
    print(trainSet)
