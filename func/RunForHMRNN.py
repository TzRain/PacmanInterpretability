import Util.EyeSightUtil as ESU
import Util.FilePathUtil as FPU
import Util.HeatMapRNN as RNNHtMp

if __name__ == '__main__':
    TrainBasePaths = "../srtp/data/dataset/test/stay"
    rnnHtMp_txt = "../srtp/data/dataset/testAns/rnnHtMp_txt.txt"
    rnnHtMp = RNNHtMp.RNNHtMpG("models/RNN_24_0.001.pth")
    fileList = FPU.getFileList(TrainBasePaths)
    writer = open(rnnHtMp_txt, "w+")
    cnt = 0
    for filePath in fileList:
        writer.write(str(rnnHtMp.draw_CAM(fileList)))
