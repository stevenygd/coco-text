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
	x,y,w,h = int(bbox[0]),int(bbox[1]),bbox[2],bbox[3]
	view = img[y:y+h,x:x+w]
	view_p = cv2.GaussianBlur(view, args['ksize'], args['sigma'])
	img_p[y:y+h,x:x+w] = view_p
	return img_p

def blackout(img, bbox, **args):
	img_p = np.copy(img)
	x,y,w,h = int(bbox[0]),int(bbox[1]),bbox[2],bbox[3]
	img_p[y:y+h,x:x+w] = 0.
	return img_p

def gen_ablation(imgId = None, mode = 'blackout', ct = None,  **args):
	"""set the text area of the given image to black,
	return (imgId, old_img, new_img) triplet"""
	if ct == None:
		ct = coco_text.COCO_Text('COCO_Text.json')
	if imgId == None:
		imgIds = ct.getImgIds(imgIds=ct.train, catIds=[('legibility','legible')])
		imgId = imgIds[np.random.randint(0,len(imgIds))]

	img = ct.loadImgs([imgId])[0]
	orig = io.imread('%s/%s/%s'%(DATA_PATH,DATA_TYPE,img['file_name']))
	annIds = ct.getAnnIds(imgIds=imgId)
	
	anns = ct.loadAnns(annIds) 

	if len(anns)==0:
		print("[WARNING] Weirdly sampled an image without text contents")

	img = orig
	for ann in anns:
		bbox = ann['bbox'] #format: [x,y,width,height]
		if mode=='blackout':
			img =  blackout(img, bbox)
		elif mode=='gaussian':
			img = gaussian(img, bbox, ksize=args['ksize'], sigma = args['sigma'])

	return imgId, orig, img


if __name__ == '__main__':
	# imgIds = ct.getImgIds(imgIds=ct.val, catIds=[('legibility','legible'),('class','machine printed')])
	# imgId = imgIds[np.random.randint(0,len(imgIds))]
	imgId, old, new = gen_ablation(mode = 'gaussian', ksize=(7,7),sigma=7.)

	print 'Saving img...'
	io.imsave(OUT_PATH+str(imgId)+'_original'+'.jpg', old)
	io.imsave(OUT_PATH+str(imgId)+'_ablated'+'.jpg', new)

