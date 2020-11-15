import pickle
import Util.HeatMapCNN as CHMG
import Util.HeatMapRNN as RHMG

if __name__ == '__main__':
    pkl_file = open("../result/trainListWithEyePos.pkl", 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()

    rnnHtMp = RHMG.RNNHtMp("../models/RNN_4_0.001.pth")
    cnnHtMp = CHMG.CNNHtMp("../models/CNN_4_0.001.pkl")

    dataList = data["dataList"]
    for item in dataList:
        path = item["path"]
        CNNPoint = (-1, -1)
        RNNPoint = (-1, -1)
        try:
            CNNPoint = cnnHtMp.draw_CAM(path)
        except Exception:
            print(Exception)
        try:
            RNNPoint = rnnHtMp.draw_CAM(path)
        except Exception:
            print(Exception)

        item["CNNPoint"] = CNNPoint
        item["RNNPoint"] = RNNPoint

    pkl_file = open("../result/trainListWithEyePosWithHeatMap.pkl", 'wb')
    pickle.dump(data, pkl_file)
    pkl_file.close()
