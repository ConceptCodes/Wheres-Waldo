import os
import cv2
import time
import argparse
import numpy as np
from tqdm import tqdm

path = os.getcwd()

weights = os.path.join(path,'weights','deploy.prototxt.txt')
models = os.path.join(path,'weights','res10_300x300_ssd_iter_140000_fp16.caffemodel')

face_detector = cv2.dnn.readNetFromCaffe(weights, models)
print('Face Detection Model Loaded Successfully!\n')

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):  return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):  return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Blur faces in images')
parser.add_argument('-i', '--image', type=str, help='image you want to blur faces on')
parser.add_argument('-v', '--video', type=str, help='videos that you want to blur faces on')
parser.add_argument('-p', '--pixels', type=str2bool, nargs='?', const=True, default=False, help='If you want the blur to be pixelated')

args = parser.parse_args()


def pixels(image, blocks=15):
	# divide the input image into NxN blocks
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, blocks + 1, dtype="int")
	ySteps = np.linspace(0, h, blocks + 1, dtype="int")
	# loop over the blocks in both the x and y direction
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			# compute the starting and ending (x, y)-coordinates
			# for the current block
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]
			# extract the ROI using NumPy array slicing, compute the
			# mean of the ROI, and then draw a rectangle with the
			# mean RGB values over the ROI in the original image
			roi = image[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(B, G, R), -1)
	# return the pixelated blurred image
	return image

if args.image is not None:
	if args.pixels:
		img = cv2.imread(args.image)
		h, w = img.shape[:2]
		kernel_width = (w // 7) | 1
		kernel_height = (h // 7) | 1
		blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
		face_detector.setInput(blob)
		output = np.squeeze(face_detector.forward())
		print('Processing image...')
		for i in tqdm(range(0, output.shape[0])):
			confidence = output[i, 2]
			if confidence > 0.4:
				box = output[i, 3:7] * np.array([w, h, w, h])
				start_x, start_y, end_x, end_y = box.astype(np.int)
				face = img[start_y: end_y, start_x: end_x]
				img[start_y: end_y, start_x: end_x] = pixels(face)
		cv2.imwrite(os.path.join(os.getcwd(),'output.jpg'), img)
		print('\nSuccess, Photo located at {}'.format(os.path.join(os.getcwd(),'output.jpg')))
	else:
		img = cv2.imread(args.image)
		h, w = img.shape[:2]

		kernel_width = (w // 7) | 1
		kernel_height = (h // 7) | 1

		blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
		face_detector.setInput(blob)
		output = np.squeeze(face_detector.forward())
		print('Processing image...')
		for i in tqdm(range(0, output.shape[0])):
			confidence = output[i, 2]
			if confidence > 0.4:
				box = output[i, 3:7] * np.array([w, h, w, h])
				start_x, start_y, end_x, end_y = box.astype(np.int)
				face = img[start_y: end_y, start_x: end_x]
				face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
				img[start_y: end_y, start_x: end_x] = face
		cv2.imwrite(os.path.join(os.getcwd(),'output.jpg'), img)
		print('Success, Photo located at {}'.format(os.path.join(os.getcwd(),'output.jpg')))

if args.video is not None:
	cap = cv2.VideoCapture(args.video)
	fourcc = cv2.VideoWriter_fourcc(*"XVID")
	_, image = cap.read()
	print(image.shape)
	out = cv2.VideoWriter("output.avi", fourcc, 20.0, (image.shape[1], image.shape[0]))
	while True:
		start = time.time()
		captured, image = cap.read()
		if not captured:
			break
		h, w = image.shape[:2]
		kernel_width = (w // 7) | 1
		kernel_height = (h // 7) | 1
		blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
		face_detector.setInput(blob)
		output = np.squeeze(face_detector.forward())
		for i in range(0, output.shape[0]):
			confidence = output[i, 2]
			if confidence > 0.4:
				box = output[i, 3:7] * np.array([w, h, w, h])
				start_x, start_y, end_x, end_y = box.astype(np.int)
				face = image[start_y: end_y, start_x: end_x]
				if args.pixels:
					image[start_y: end_y, start_x: end_x] = pixels(face)
				else:
					face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
					image[start_y: end_y, start_x: end_x] = face
		cv2.imshow("image", image)
		if cv2.waitKey(1) == ord("q"):
			break
		time_elapsed = time.time() - start
		fps = 1 / time_elapsed
		print("FPS:", fps)
		out.write(image)

	cv2.destroyAllWindows()
	cap.release()
	out.release()

