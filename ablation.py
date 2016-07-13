import coco_text
import coco_evaluation

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

import cv2

import os 
CD = os.path.dirname(os.path.realpath(__file__))

print CD

DATA_PATH = CD + '/../data/coco/'
DATA_TYPE = 'train2014'
OUT_PATH = 'result/'

def gaussian(img, bbox, **args):
    """ [args]: ksize - (h,w) tuple specifying kernel size"""
    img_p = np.copy(img)
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    view = img[y:y+h,x:x+w]
    view_p = cv2.GaussianBlur(view, args['ksize'], args['sigma'])
    img_p[y:y+h,x:x+w] = view_p
    return img_p

def blackout(img, bbox, **args):
    img_p = np.copy(img)
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    img_p[y:y+h,x:x+w] = 0.
    return img_p

def gen_ablation(imgIds = [], mode = 'blackout', ct = None,  **args):
    """Perform specified ablation on every image specified by the imgIds list.
    If no imgId is specified, will randomly sample an image with text.
    return (imgId, old_img, new_img) list"""
    if ct == None:
        ct = coco_text.COCO_Text('COCO_Text.json')
    if imgIds == []:
        imgIds = ct.getImgIds(imgIds=ct.train, catIds=[('legibility','legible')])
        imgIds = [imgIds[np.random.randint(0,len(imgIds))]]

    imgs = ct.loadImgs(imgIds)
    results = []
    for img in imgs:
        orig = io.imread('%s/%s/%s'%(DATA_PATH,DATA_TYPE,img['file_name']))
        annIds = ct.getAnnIds(imgIds=img['id'])
        
        anns = ct.loadAnns(annIds) 

        if len(anns)==0:
            print("[WARNING] Weirdly sampled an image without text contents:{}".format(img['file_name']))

        running = orig
        for ann in anns:
            bbox = ann['bbox'] #format: [x,y,width,height]
            if mode=='blackout':
                running = blackout(running, bbox)
            elif mode=='gaussian':
                running = gaussian(running, bbox, ksize=args['ksize'], sigma = args['sigma'])
        results.append((img['id'], orig, running))
    return results


if __name__ == '__main__':
    # imgIds = ct.getImgIds(imgIds=ct.val, catIds=[('legibility','legible'),('class','machine printed')])
    # imgId = imgIds[np.random.randint(0,len(imgIds))]
    results = gen_ablation(mode = 'gaussian', ksize=(7,7),sigma=7.)

    for imgId, old, new in results:
        print 'Saving img {}'.format(imgId)
        io.imsave(OUT_PATH+str(imgId)+'_original'+'.jpg', old)
        io.imsave(OUT_PATH+str(imgId)+'_ablated'+'.jpg', new)
