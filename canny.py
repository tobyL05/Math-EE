import cProfile
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
	#cv2.imwrite('2dblurred.jpg',output)
	return output

def sepblur(img):
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

def sobel(gray_img):
	output = np.zeros_like(gray_img)
	angles = np.zeros_like(gray_img)
	horiimg = np.zeros_like(gray_img)
	vertimg = np.zeros_like(gray_img)
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
			angle = np.rad2deg(np.arctan2(vert,hori))
			if angle < 0:
				angle += 180
			angles[i-1,j-1] = angle
			output[i-1,j-1] = np.sqrt(np.square(hori) + np.square(vert))

	#cv2.imwrite('sobel.jpg',output)
	return (output,angles)

def nms(img,angles):
	nmsimg = np.zeros_like(img)
	rows = img.shape[0]
	cols = img.shape[1]
	angle = 0
	for i in range(1,rows-1):
		for j in range(1,cols-1):
			angle = angles[i-1,j-1]
			#print(f"{(i-1,j-1)}: {angles[i-1,j-1]}")
			if -22.5 <= angle <= 22.5 or (angle >= 157.5 or angle <= -157.5):
				if img[i-1,j-1] < img[i-1,j] or img[i-1,j-1] < img[i-1,j-2]:
					nmsimg[i-1,j-1] = 0
				else:
					nmsimg[i-1,j-1] = img[i-1,j-1]
			elif 22.5 <= angle <= 67.5 or -157.5 <= angle <= -112.5:
				if img[i-1,j-1] < img[i,j] or img[i-1,j-1] < img[i-2,j-2]:
					nmsimg[i-1,j-1] = 0
				else:
					nmsimg[i-1,j-1] = img[i-1,j-1]
			elif -67.5 <= angle <= -22.5 or 112.5 <= angle <= 157.5:
				if img[i-1,j-1] < img[i-2,j] or img[i-1,j-1] < img[i,j-2]:
					nmsimg[i-1,j-1] = 0
				else:
					nmsimg[i-1,j-1] = img[i-1,j-1]
			elif 67.5 <= angle <= 112.5 or -112.5 <= angle <= -67.5:
				if img[i-1,j-1] < img[i,j-1] or img[i-1,j-1] < img[i-2,j-1]:
					nmsimg[i-1,j-1] = 0
				else:
					nmsimg[i-1,j-1] = img[i-1,j-1]
			#else:
				#if img[i-1,j-1] < img[i-2,j] and img[i-1,j-1] < img[i,j-2]:
					#img[i-1,j-1] = 0
	
	
	#cv2.imwrite('sobelNMS.jpg',nmsimg)
	return nmsimg

def doublethresh(img):
	thresholded = np.zeros_like(img)
	low = 50
	high = 125
	rows = img.shape[0]
	cols = img.shape[1]
	for i in range(1, rows-1):
		for j in range(1,cols-1):
			if img[i-1,j-1] <= low:
				thresholded[i-1,j-1] = 0
			elif img[i-1,j-1] > low and img[i-1,j-1] < high:
				thresholded[i-1,j-1] = 150
			else:
				thresholded[i-1,j-1] = 255
	#jcv2.imwrite('thresholded.jpg',thresholded)
	return thresholded

def hysteresis(img):
	strong = np.zeros_like(img)
	rows = img.shape[0]
	cols = img.shape[1]
	topr=0
	topl=0
	topm=0
	l=0
	r=0
	botr=0
	botl=0
	botm=0 
	neighbors = []
	for i in range(1, rows -1):
		for j in range(1, cols -1):
			topr = img[i,j]
			topl = img[i,j-2]
			topm = img[i,j-1]
			l = img[i-1,j-2]
			r = img[i-1,j]
			botr = img[i-2,j]
			botl = img[i-2,j-2]
			botm = img[i-2,j-1]
			neighbors =[topr,topl,topm,l,r,botr,botl,botm]
			if img[i-1,j-1] == 150:
				if any(n >= 125 for n in neighbors):
					strong[i-1,j-1] = 255
					img[i-1,j-1] = 255
				else:
					strong[i-1,j-1] = 0
					img[i-1,j-1] = 0
			elif img[i-1,j-1] == 255:
				strong[i-1,j-1] = 255

	#cv2.imwrite('final.jpg',strong)

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

def cannyspeedtest(func):
	time = 0
	sum = 0 														#to hold the sum
	for i in range(5): 												#convolve 5 times
		time += cannyPerformance(func)
		sum += time
		print(f"Trial {i+1} completion time: {time}")  #output after each convolution
	print(f"Average completion time (5 trials): {sum/5}")
	#return sum/5 													#return the average

def cannyPerformance(func):
	start_time = time.time()
	func()
	return round(time.time() - start_time,3) 

def canny():
	imgpath = 'nztowerds.jpg' #original
	#imgpath = 'mount.JPG'

	#print("Grayscaling...")
	grayimg = grayscale(cv2.imread(imgpath)) #grayscale

	#print("Blurring...")
	img = sepblur(grayimg) #blur
	#img = opencvgaussian(grayimg) #much faster

	#print("Getting gradient magnitude and direction...")
	grad = sobel(img) #grad magnitude and dir

	#print("NMS...")
	nmsimg = nms(grad[0],grad[1]) #nms

	#print("Thresholding...")
	thresh = doublethresh(nmsimg) #double thresholding

	#print("Hysteresis...")
	hysteresis(thresh) #hysteresis

def cv2Canny():
	imgpath = 'nztowerds.jpg'
	img = cv2.imread(imgpath)
	edges = cv2.Canny(img,50,125)
	#cv2.imwrite("cv2CannyDS.jpg",edges)

def sobeldet():
	imgpath = 'nztowerds.jpg'
	img = cv2.imread(imgpath)
	img = sepblur(grayscale(img))
	det = sobel(img)
	edges = det[0]
	angles = det[1]
	cv2.imwrite('angletest.jpg',nms(edges,angles))
	#cv2.imwrite("exout.png",edges)

def start():
	sobeldet()
	#cannyspeedtest(canny)
	#cProfile.run('canny()')
	#cv2Canny()

if __name__ == "__main__":
	start()




