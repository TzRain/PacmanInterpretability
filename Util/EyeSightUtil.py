import os
import numpy as np
from PIL import Image


def checkWhite(r, b, g):
    return r > 170 and b > 170 and g > 200


def checkWhiteYellow(r, b, g):
    return r > 200 and g > 200 and b > 40


def GetEyePos(inputPath):
    global img, White
    img = Image.open(inputPath)
    width = img.size[0]
    height = img.size[1]
    White = np.zeros(img.size)
    for i in range(0, width):
        for j in range(0, height):
            r, g, b = (img.getpixel((i, j)))
            if checkWhite(r, b, g) or checkWhiteYellow(r, b, g):
                White[i][j] = True
    mxI = 0
    mxJ = 0
    mxR = 0
    for i in range(0, width):
        for j in range(0, height):
            limit = min(width - i, height - j, i + 1, j + 1)
            for r in range(mxR, limit):
                if not White[i][j + r]:
                    break
                if not White[i + r][j]:
                    break
                if not White[i][j - r]:
                    break
                if not White[i - r][j]:
                    break
                if r > mxR:
                    mxR = r
                    mxI = i
                    mxJ = j
    return mxI, mxJ, mxR, width, height
