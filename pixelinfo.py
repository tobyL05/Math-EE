from re import I
import cv2 
import numpy as np

def getcolorpixel(img):
	for i in range(10):
		print(f"Pixel at ({i}, 0): {np.flip(img[i][0])}")

def getgraypixel(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	for i in range(10):
		print(f"Pixel at ({i}, 0): {img[i][0]}")
	cv2.imwrite('nztowergray.jpg',img)

def getimgmatrix(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	matr = np.array(img)
	matr.astype(int)
	np.set_printoptions(suppress=True)
	#np.savetxt("imagematrix.txt",matr,fmt='%-3d')
	#for i in range(50):
		#for j in range(50):
	return matr

def getpixelinfos(img):
	print("Original image")
	getcolorpixel(img)

	print("\n")
	
	print("Grayscaled image")
	getgraypixel(img)

def start():
	imgpath = "nztowerds.jpg"
	img = cv2.imread(imgpath)
	print(getimgmatrix(img))

if __name__ == "__main__":
	start()