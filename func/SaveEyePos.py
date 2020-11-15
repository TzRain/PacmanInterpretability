import pickle
import Util.EyeSightUtil as ESU

if __name__ == '__main__':
    pkl_file = open("../result/trainListWithEyePos.pkl", 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()

    dataList = data["dataList"]
    i = 0
    cnt = 0
    for item in dataList:
        if i < data["eyePosLength"]:
            i += 1
            continue
        path = item["path"]
        item["eyePos"] = ESU.GetEyePos(path)
        cnt += 1
        i += 1
        print(cnt)
        if cnt == 100:
            break
    data["eyePosLength"] = data["eyePosLength"] + cnt
    writer = open("../result/trainListWithEyePos.pkl", "wb")
    pickle.dump(data, writer)
    writer.close()
