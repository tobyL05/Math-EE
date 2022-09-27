from re import M
import time
import numpy as np
import cv2
  
def grayscale(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#cv2.imwrite('grayscale.jpg',img)
	return img
	
def blur(img):
	output = np.zeros_like(img)
	#img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_WRAP) #Handle edge cases (wrap around)
	gaussiankernel = np.array([
		[1,2,1],
		[2,4,2],
		[1,2,1]
	])/16 #Gaussian kernel divided by 16
	gaussiankernel = np.flipud(np.fliplr(gaussiankernel)) #transpose gaussian kernel

	rows = img.shape[0]
	cols = img.shape[1]

	for r in range(rows):
		for c in range(cols):
			for m in range(3):
				mm = 2-m
				for n in range(3):
					nn = 2-n
					ii = r + (2-mm)
					jj = c + (2-nn)
					if ii>=0 and ii < rows and jj >=0 and jj < cols:
						output[r][c] += img[ii][jj] * gaussiankernel[mm][nn]
	cv2.imwrite('2dblurred.jpg',output)

def sepblur(img):
	#img = cv2.transpose(img)
	output = np.zeros_like(img)
	output2 = np.zeros_like(img)
	kernelc = np.array([[1],[2],[1]])/4
	kernelr = np.array([1,2,1])/4 #[row][column]
	rows = img.shape[0]
	cols = img.shape[1]
	for r in range(rows):
		for c in range(cols):
			for kc in range(3):
				ii = c - kc
				if ii>=0 and ii < cols:
					output[r][c] += kernelc[kc][0] * img[r][ii]

	for r in range(rows):
		for c in range(cols):
			for kr in range(3):
				jj = r - kr
				if jj>=0 and jj < rows:
					#output[r][c] += kernelr[kr] * img[jj][c]
					output2[r][c] += kernelr[kr] * output[jj][c]

	cv2.imwrite('sepblurred.jpg',output2)

def start():
	imgpath = 'nztowerds.jpg'
	grayimg = grayscale(cv2.imread(imgpath))
	#blur(grayimg)
	sepblur(grayimg)
	#cv2.imwrite('opencvblurred.jpg',cv2.GaussianBlur(grayimg,(3,3),0))
	
if __name__ == "__main__":
	start_time = time.time()
	start()
	print(f"time in seconds:  {round(time.time() - start_time,3)} seconds")




