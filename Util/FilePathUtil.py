import os
import pickle

def AddPostfix(path, pf):
    pathList = path.rsplit(".", 1)
    if len(pathList) == 1:
        return path + pf
    return pathList[0] + pf + "." + pathList[1]


def ReName(rpath):
    return rpath.replace('\\', '/')


def getFileList(basePath, addBasePath=True):
    files = os.listdir(basePath)
    fileList = []
    if not addBasePath:
        return files
    for file in files:
        fileList.append(basePath.rstrip('/') + '/' + file)
    return fileList


def isImage(path):
    pathList = path.rsplit(".", 1)
    if len(pathList) == 1:
        return False
    if pathList[1] == "jpg" or pathList[1] == "png":
        return True
    return False


# IO 0 表示创建
# IO 1 表示添加在尾部
# 读取 basePath 下所有的文件 并打上label
# list = [APath+label ,APath+label，APath+label ...]
def createPathListFile(BPath, label):
    files = os.listdir(BPath)
    res = []
    for path in files:
        data = {
            "path": BPath+'/'+path,
            "label": label
        }
        res.append(data)
    return res


# dic = {'stay': 0, 'left': 1, 'up': 2, 'right': 3, 'down': 4}

def createList(basePath, outputPath, OW=False):
    res = {
        "dataList": []
    }
    writer = open(outputPath, "wb")
    res["dataList"] = createPathListFile(basePath + "stay", "0") + createPathListFile(basePath + "left", "1") + \
                      createPathListFile(basePath + "up", "2") + createPathListFile(basePath + "right", "3") + \
                      createPathListFile(basePath + "down", "4")
    pickle.dump(res, writer)
    writer.close()


if __name__ == '__main__':
    # createList("../../srtp/data/dataset/train/", "../result/trainList.pkl", True)
    # pkl_file = open("../result/trainList.pkl", 'rb')
    # data = pickle.load(pkl_file)
    # pkl_file.close()

    createList("../../srtp/data/dataset/test/", "../result/testList.pkl", True)
    pkl_file = open("../result/testList.pkl", 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()