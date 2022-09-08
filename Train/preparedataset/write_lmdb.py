import os
import lmdb # install lmdb by "pip install lmdb"
from PIL import Image
import six
import glob

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    buf = six.BytesIO()
    buf.write(imageBin)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    imgH, imgW = img.size
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def createDataset(outputPath, imagePathList, checkValid=True):
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
            if checkValid:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            imageKey = 'image-%09d' % cnt#9位数不足填零
            cache[imageKey] = imageBin
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == "__main__":
    lmdb_output_path = './trainingset'
    imagePath = './cropped_training_set'
    imageList = glob.glob(os.path.join(imagePath,'*.png'))
    createDataset(lmdb_output_path, imageList)