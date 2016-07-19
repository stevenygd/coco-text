import coco_text
import coco_evaluation

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

import cv2

import os 
CD = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0,CD+'/../../coco/PythonAPI/')
from pycocotools.coco import COCO, mask

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

def median(img, bbox, **args):
    """[args]: width - integer specifying dimension of square window"""
    img_p = np.copy(img)
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    view = img[y:y+h,x:x+w]
    view_p = cv2.medianBlur(view, args['width'])
    img_p[y:y+h,x:x+w] = view_p
    return img_p

def destroy_bg(img, imgId, coco):
    """Blackout everything in the background that is not annotated
    as a coco instance"""
    annIds = coco.getAnnIds(imgIds=imgId)
    anns = coco.loadAnns(annIds)

    segs = []
    for ann in anns:
        segs += ann['segmentation']

    mk = mask.merge( mask.frPyObjects(segs, img.shape[0], img.shape[1]), intersect = 0)
    mk = mask.decode([mk])
    return img*mk


def ablate(imgIds = [], mode ='destroy', coco = None, ct = None,  **args):
    """[ablation entry point 2.0]
    Created to accomodate background-destroying ablation. Will dispatch all 
    old ablations (gaussian, blackout, & median) to gen_ablation."""

    if ct is None:
        ct = coco_text.COCO_Text(os.path.join(CD, 'COCO_Text.json'))
    if imgIds == []:
        imgIds = ct.getImgIds(imgIds=ct.train, catIds=[('legibility','legible')])
        imgIds = [imgIds[np.random.randint(0,len(imgIds))]]

    #dispatch to old ablation entry point
    if not mode == 'destroy':
        return gen_ablation(imgIds, mode, ct, **args)

    #else do destroy_bg
    if coco is None:
        coco = COCO('%s/annotations/instances_%s.json'%(DATA_PATH,DATA_TYPE))
    imgs = ct.loadImgs(imgIds)
    results = []
    for idx, img in enumerate(imgs):
        print("Ablating image {}/{}".format(idx+1, len(imgIds)))
        orig = io.imread('%s/%s/%s'%(DATA_PATH,DATA_TYPE,img['file_name']))
        ablt = destroy_bg(orig, img['id'], coco)
        results.append((img['id'], orig, ablt))
    return results


def gen_ablation(imgIds = [], mode = 'blackout', ct = None,  **args):
    """Perform specified ablation on every image specified by the imgIds list.
    If no imgId is specified, will randomly sample an image with text.
    return (imgId, old_img, new_img) list"""
    imgs = ct.loadImgs(imgIds)
    results = []
    for idx, img in enumerate(imgs):
        print("Ablating image {}/{}".format(idx+1, len(imgIds)))
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
            elif mode=='median':
                running = median(running, bbox, width=args['width'])
        results.append((img['id'], orig, running))
    return results


if __name__ == '__main__':
    # imgIds = ct.getImgIds(imgIds=ct.val, catIds=[('legibility','legible'),('class','machine printed')])
    # imgId = imgIds[np.random.randint(0,len(imgIds))]
    results = ablate( mode = 'destroy', width=7)

    for imgId, old, new in results:
        print 'Saving img {}'.format(imgId)
        io.imsave(OUT_PATH+str(imgId)+'_original'+'.jpg', old)
        io.imsave(OUT_PATH+str(imgId)+'_ablated'+'.jpg', new)
