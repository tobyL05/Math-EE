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

def prewitt(gray_img):
	output = np.zeros_like(gray_img)
	horizontal = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
	vertical = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	rows = gray_img.shape[0]
	cols = gray_img.shape[1]
	hori,vert = 0,0
	for i in range(1,rows-1):
		for j in range(1,cols-1):
			hori,vert = 0,0
			for kr in range(3):
				for kc in range(3):
					hori += horizontal[kr][kc] * gray_img[i-kr][j-kc]
					vert += vertical[kr][kc] * gray_img[i-kr][j-kc]
					#print(hori,vert)
			#hori = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                    #(horizontal[0, 1] * gray_img[i - 1, j]) + \
                    #(horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                    #(horizontal[1, 0] * gray_img[i, j - 1]) + \
                    #(horizontal[1, 1] * gray_img[i, j]) + \
                    #(horizontal[1, 2] * gray_img[i, j + 1]) + \
                    #(horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                    #(horizontal[2, 1] * gray_img[i + 1, j]) + \
                    #(horizontal[2, 2] * gray_img[i + 1, j + 1])

			#vert= (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                #(vertical[0, 1] * gray_img[i - 1, j]) + \
                #(vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                #(vertical[1, 0] * gray_img[i, j - 1]) + \
                #(vertical[1, 1] * gray_img[i, j]) + \
                #(vertical[1, 2] * gray_img[i, j + 1]) + \
                #(vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                #(vertical[2, 1] * gray_img[i + 1, j]) + \
                #(vertical[2, 2] * gray_img[i + 1, j + 1])

    		# Edge Magnitude
			mag = np.sqrt(pow(hori, 2.0) + pow(vert, 2.0))
			output[i - 1, j - 1] = mag

	cv2.imwrite('prewitt.jpg',output)
	#cv2.imshow('prewitt',output)
	#cv2.waitKey(0)


def sepprewitt(img):
	hori = np.zeros_like(img)
	temphori = np.zeros_like(img)
	vert = np.zeros_like(img)
	tempvert = np.zeros_like(img)
	out = np.zeros_like(img)
	boxc = np.array([[1],[1],[1]])/3
	prewr = np.array([1,0,-1])
	boxr = np.array([1,1,1])/3
	prewc = np.array([[1],[0],[-1]])
	rows = img.shape[0]
	cols = img.shape[1]

	for r in range(rows):
		for c in range(cols):
			temp = 0
			for kc in range(3):
				ii = c - kc
				temp += boxc[kc][0] * img[r][ii]
			temphori[r][c] = abs(temp)

	cv2.imwrite('prewittboxX.jpg',temphori)
					
	for r in range(rows):
		for c in range(cols):
			temp = 0
			for kr in range(3):
				jj = r - kr
				temp += prewr[kr] * temphori[jj][c]
			hori[r][c] = abs(temp)

	cv2.imwrite('hori.jpg',hori)

	for r in range(rows):
		for c in range(cols):
			temp = 0
			for kr in range(3):
				jj = r - kr
				temp += boxr[kr] * img[jj][c]
			tempvert[r][c] = abs(temp)

	cv2.imwrite('prewittboxY.jpg',tempvert)

	for r in range(rows):
		for c in range(cols):
			temp = 0
			for kc in range(3):
				ii = c - kc
				temp += prewc[kc][0] * tempvert[r][ii]
			vert[r][c] = abs(temp)

	
	cv2.imwrite('vert.jpg',vert)

	for r in range(rows):
		for c in range(cols):
			out[r][c] = np.sqrt(pow(hori[r][c],2) + pow(vert[r][c],2))

	cv2.imwrite('sep_prewitt.jpg',out)
	#cv2.imwrite('prewittadd.jpg',hori + vert)
	#cv2.imshow('sep prewitt',out)
	#cv2.waitKey(0)

def opencvgaussian(img):
	cv2.imwrite('opencvgaussian.jpg',cv2.GaussianBlur(img, (3,3),0))

def speedtest(func,img):
	sum = 0 														#to hold the sum
	for i in range(5): 												#convolve 5 times
		start_time = time.time()									#for each convolution, get starting time
		func(img) 													#separable/non-separable convolution
		sum += round(time.time() - start_time,3) 					#subtract current time from starting time
		print(f"Trial {i+1}: {round(time.time() - start_time,3)}")  #output after each convolution
	return sum/5 													#return the average

def start():
	imgpath = 'nztowerds.jpg'
	grayimg = grayscale(cv2.imread(imgpath))
	#img = blur(grayimg)
	img = sepblur(grayimg)
	prewitt(img)
	#sepprewitt(img)
	#cv2prewitt(img)
	#cv2.imwrite('opencvblurred.jpg',cv2.GaussianBlur(grayimg,(3,3),0))
	#print(f"average time in seconds (5 trials):  {speedtest(opencvgaussian,grayimg)} seconds")
	
if __name__ == "__main__":
	start()




