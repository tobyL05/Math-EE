import math
import time
import numpy as np
import cv2
  
def grayscale(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#cv2.imwrite('grayscale.jpg',img)
	return img
	
def blur(img):
	output = np.zeros_like(img)
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
	return output

def sepblur(img):
	print("blurring")
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
					output2[r][c] += kernelr[kr] * output[jj][c]

	#cv2.imwrite('sepblurred.jpg',output2)
	return output2

def cv2prewitt(img):
	output = np.zeros_like(img)
	kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
	kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	img_prewittx = cv2.filter2D(img, -1, kernelx)
	img_prewitty = cv2.filter2D(img, -1, kernely)
	#cv2.imwrite('opencv2prewittx.jpg',img_prewittx)
	#cv2.imwrite('opencv2prewitty.jpg',img_prewitty)
	for r in range(img.shape[0]):
		for c in range(img.shape[1]):
			output[r][c] = np.sqrt(pow(img_prewittx[r][c],2) + pow(img_prewitty[r][c],2))

	cv2.imwrite('cv2prewitt.jpg',output)

def sobel(gray_img):
	output = np.zeros_like(gray_img)
	angles = np.zeros_like(gray_img)
	horizontal = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	vertical = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
	rows = gray_img.shape[0]
	cols = gray_img.shape[1]
	for i in range(1,rows-1):
		for j in range(1,cols-1):
			hori,vert = 0,0
			for kr in range(3):
				for kc in range(3):
					hori += horizontal[kr][kc] * gray_img[i-kr][j-kc]
					vert += vertical[kr][kc] * gray_img[i-kr][j-kc]

    		# Edge Magnitude
			mag = np.sqrt(pow(hori, 2.0) + pow(vert, 2.0))
			output[i-1,j-1] = mag
			angles[i-1,j-1] = np.arctan2(vert,hori)
			#if mag > 100:
				#output[i-1,j-1] = mag

	cv2.imwrite('sobel.jpg',output)
	return output,angles

def nms(img,angles):
	rows = img.shape[0]
	cols = img.shape[1]
	for i in range(1,rows-1):
		for j in range(1,cols-1):
			print(f"{(i-1,j-1)}: {angles[i-1,j-1]}")
			if angles[i-1,j-1] < 0:
				break
			#if -22.5 <= gradnorm <= 22.5 or -157.5 <= gradnorm <= 157.5:
				#if output[i-2,j-1] > output[i-1,j-1] or output[i,j-1] > output[i-1,j-1]:
					#output[i-1,j-1] = 0
			#elif 67.5 <= gradnorm <= 122.5 or -122.5<= gradnorm <= -67.5:
				#if output[i-1,j] > output[i-1,j-1] or output[i-1,j-2] > output[i-1,j-1]:
					#output[i-1,j-1] = 0 
			#elif 22.5 <= gradnorm <= 67.5 or -157.5 <= gradnorm <= -122.5:
				#if output[i-1,j-1] < output[i,j] or output[i-1,j-1] < output[i-2,j-2]:
					#output[i-1,j-1] = 0
			#elif 122.5 <= gradnorm <= 157.5 or -67.5 <= gradnorm <= -22.5:
				#if output[i-1,j-1] < output[i-2,j] or output[i-1,j-1] < output[i,j-2]:
					#output[i-1,j-1] = 0


def sepprewitt(img):
	hori = np.zeros_like(img)
	temphori = np.zeros_like(img)
	vert = np.zeros_like(img)
	tempvert = np.zeros_like(img)
	out = np.zeros_like(img)
	normx = np.zeros_like(img)
	normy = np.zeros_like(img)
	blurc = np.array([[1],[2],[1]])/4
	sobelr = np.array([-1,0,1])
	blurr = np.array([1,2,1])/4
	sobelc = np.array([[-1],[0],[1]])
	rows = img.shape[0]
	cols = img.shape[1]

	for r in range(rows):
		for c in range(cols):
			temp = 0
			for kc in range(3):
				ii = c - kc
				temp += blurc[kc][0] * img[r][ii]
			temphori[r][c] = abs(temp)

	cv2.imwrite('gaussX.jpg',temphori)
					
	for r in range(rows):
		for c in range(cols):
			temp = 0
			for kr in range(3):
				jj = r - kr
				temp += sobelr[kr] * temphori[jj][c]
			normx[r][c] = temp
			hori[r][c] = abs(temp)

	cv2.imwrite('sobelX.jpg',hori)

	for r in range(rows):
		for c in range(cols):
			temp = 0
			for kr in range(3):
				jj = r - kr
				temp += blurr[kr] * img[jj][c]
			tempvert[r][c] = abs(temp)

	cv2.imwrite('gaussY.jpg',tempvert)

	for r in range(rows):
		for c in range(cols):
			temp = 0
			for kc in range(3):
				ii = c - kc
				temp += sobelc[kc][0] * tempvert[r][ii]
			normy[r][c] = temp
			vert[r][c] = abs(temp)
	
	cv2.imwrite('sobelY.jpg',vert)
	i = 0
	for r in range(rows):
		for c in range(cols):
			out[r][c] = np.sqrt(pow(hori[r][c],2) + pow(vert[r][c],2))
			gradnorm = np.arctan2(float(normy[r][c]),float(normx[r][c])) * (180/np.pi)
			#gradnorm = round(gradnorm,2)
			print(f"{(r,c)}Gx = {normx[r][c]}, Gy = {normy[r][c]}")
			#if gradnorm <= 22.5 and gradnorm >= -22.5: #vertical edge
			#print(f"Grad dir at {r,c}: {gradnorm}")
			#if i > 100:
				#break
			#else:
				#i += 1

	cv2.imwrite('sobel.jpg',out)
	#cv2.imwrite('prewittadd.jpg',hori + vert)
	#cv2.imshow('sep prewitt',out)
	#cv2.waitKey(0)

def opencvgaussian(img):
	return cv2.GaussianBlur(img, (5,5),0)
	#cv2.imwrite('opencvgaussian.jpg',cv2.GaussianBlur(img, (5,5),0))
	

def speedtest(func,img):
	sum = 0 														#to hold the sum
	for i in range(5): 												#convolve 5 times
		start_time = time.time()									#for each convolution, get starting time
		func(img) 													#separable/non-separable convolution
		sum += round(time.time() - start_time,3) 					#subtract current time from starting time
		print(f"Trial {i+1}: {round(time.time() - start_time,3)}")  #output after each convolution
	return sum/5 													#return the average

def start():
	imgpath = 'nztowerds.jpg' #downscaled
	#imgpath = 'nztowerORI.jpg' #original
	grayimg = grayscale(cv2.imread(imgpath))
	img = sepblur(grayimg)
	img, angles = sobel(img)
	nms(img, angles)

	#cv2.imwrite('opencvblurred.jpg',cv2.GaussianBlur(grayimg,(3,3),0))
	#print(f"average time in seconds (5 trials):  {speedtest(opencvgaussian,grayimg)} seconds")
	
if __name__ == "__main__":
	start()




