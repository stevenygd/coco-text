import coco_text
import coco_evaluation

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

import cv2

DATA_PATH = '/Users/zhuyifan/Downloads/coco/'
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

def gen_ablation(saveto, imgId = None, mode = 'blackout', **args):
	"""set the text area of the given image to black,
	save both the original and ablated image to [saveto]"""
	ct = coco_text.COCO_Text('COCO_Text.json')

	if imgId == None:
		imgIds = ct.getImgIds(imgIds=ct.train, catIds=[('legibility','legible')])
		imgId = imgIds[np.random.randint(0,len(imgIds))]

	img = ct.loadImgs([imgId])[0]
	I = io.imread('%s/%s/%s'%(DATA_PATH,DATA_TYPE,img['file_name']))
	annIds = ct.getAnnIds(imgIds=imgId)
	bbox = ct.loadAnns(annIds)[0]['bbox'] #format: [x,y,width,height]

	print 'Saving img...'
	io.imsave(saveto+str(imgId)+'_original'+'.jpg', I)
	
	if mode=='blackout':
		I_p =  blackout(I, bbox)
	elif mode=='gaussian':
		I_p = gaussian(I, bbox, ksize=args['ksize'], sigma = args['sigma'])

	io.imsave(saveto+str(imgId)+'_ablated'+'.jpg', I_p)


if __name__ == '__main__':
	ct = coco_text.COCO_Text('COCO_Text.json')
	# imgIds = ct.getImgIds(imgIds=ct.val, catIds=[('legibility','legible'),('class','machine printed')])
	# imgId = imgIds[np.random.randint(0,len(imgIds))]
	gen_ablation(OUT_PATH, mode = 'gaussian', ksize=(7,7),sigma=7.)

