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
sys.path.insert(0,CD+'/../coco/PythonAPI/')
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
    if not len([p for p in view.shape if p > 0]) == len(view.shape):
       return img_p

    view_p = cv2.medianBlur(view, args['width'])
    img_p[y:y+h,x:x+w] = view_p
    return img_p

def destroy_bg(img, imgId, coco):
    """Blackout everything in the background that is not annotated
    as a coco instance
    [pre] Every imgid passed in has at least one object instance annotated."""
    annIds = coco.getAnnIds(imgIds=imgId)
    anns = coco.loadAnns(annIds)

    h, w = img.shape[0], img.shape[1]
    segs, rles = [], []
    for ann in anns:
        if not ann['iscrowd']:
            segs += ann['segmentation']
        else:
            rles.append(ann['segmentation'])

    mk = None
    if len(segs)!=0:
        mk = mask.merge( mask.frPyObjects(segs, h,w), intersect = 0)
    if len(rles)!=0:
        mk2 = mask.merge( mask.frPyObjects(rles, h,w), intersect = 0)
        mk = mask.merge([mk,mk2], intersect = 0) if mk!=None else mk2
    mk = mask.decode([mk])

    #handle non-RGB images
    if len(img.shape)<3:
        return img[:,np.newaxis]*mk
    else:
        return img*mk


def ablate(imgIds = [], mode ='destroy', out_path="tmp", coco = None, ct = None,  **args):
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
        return gen_ablation(imgIds, mode, ct, out_path=out_path, **args)

    #else do destroy_bg
    if coco is None:
        coco = COCO('%s/annotations/instances_%s.json'%(DATA_PATH,DATA_TYPE))
    imgs = coco.loadImgs(imgIds)
    results = []
    for idx, img in enumerate(imgs):
        print("Ablating image {}/{} with id {} ".format(idx+1, len(imgIds), img['id']))
        ori_file_name = os.path.join(CD, DATA_PATH, DATA_TYPE, img['file_name'])
        orig = io.imread(ori_file_name)

        ablt = destroy_bg(orig, img['id'], coco)
        out_file_name = os.path.join(CD, "..", out_path, "%s_%s"%(mode, img['file_name']))
        io.imsave(out_file_name, ablt)

        results.append((img['id'], ori_file_name, out_file_name))
    return results


def gen_ablation(imgIds = [], mode = 'blackout', ct = None, out_path="tmp", **args):
    """Perform specified ablation on every image specified by the imgIds list.
    If no imgId is specified, will randomly sample an image with text.
    return (imgId, old_img, new_img) list"""
    imgs = ct.loadImgs(imgIds)
    results = []
    for idx, img in enumerate(imgs):
        print("Ablating image {}/{}".format(idx+1, len(imgIds)))
        ori_file_name = '%s/%s/%s'%(DATA_PATH,DATA_TYPE,img['file_name'])
        orig = io.imread(ori_file_name)
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
        out_file_name = os.path.join(CD, "..", out_path, "%s_%s"%(mode, img['file_name']))
        io.imsave(out_file_name, running)
        results.append((img['id'], ori_file_name, out_file_name))
    return results


if __name__ == '__main__':
    # imgIds = ct.getImgIds(imgIds=ct.val, catIds=[('legibility','legible'),('class','machine printed')])
    # imgId = imgIds[np.random.randint(0,len(imgIds))]
    from six.moves import cPickle as pkl
    with open('../input/no_text_has_instance_img_ids') as f:
        imgIds = pkl.load(f)

    results = ablate( imgIds = imgIds, mode = 'destroy', width=7)

    for imgId, old, new in results:
        print 'Saving img {}'.format(imgId)
        io.imsave(OUT_PATH+str(imgId)+'_original'+'.jpg', old)
        io.imsave(OUT_PATH+str(imgId)+'_ablated'+'.jpg', new)
