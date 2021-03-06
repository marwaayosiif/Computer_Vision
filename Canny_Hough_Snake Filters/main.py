from PyQt5 import QtWidgets, QtGui
from imageview import Ui_MainWindow
import sys
import numpy as np
import cv2
from scipy.ndimage.filters import convolve
QPixmap = QtGui.QPixmap
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import matplotlib.pyplot as plt
from collections import defaultdict
import snake as snake

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.image = None
        self.snake = None
        self.n = 3
        self.precent = 0.1
        self.alpha = 0.8
        self.minIntensity = 0
        self.maxIntensity = 255
        self.Th = 150
       
        self.ui.menuExit.triggered.connect(exit)
        self.ui.loadEdgeImg.clicked.connect(lambda: self.getPictures(5))
        self.ui.CannyEdgeBtn.clicked.connect(lambda :self.canny(0))
        self.ui.houghLineBtn.clicked.connect(self.line)
        self.ui.houghCircleBtn.clicked.connect(self.circle)
        self.ui.LoadTab5.clicked.connect(lambda:self.getPictures(6))
        self.ui.Apply.clicked.connect(self.snakeAlgorithm)
        self.ui.Reset.clicked.connect(self.clear)
      
        self.ui.groupBox_2.hide()
        self.ui.selectTab1.activated.connect(self.chooseFilter)
        self.ui.menuExit.triggered.connect(exit)
        self.ui.loadTab1.clicked.connect(lambda: self.getPictures(1))
        self.ui.loadTab2.clicked.connect(lambda: self.getPictures(2))
        self.ui.load1Tab3.clicked.connect(lambda: self.getPictures(3))
        self.ui.load2tab3.clicked.connect(lambda: self.getPictures(4))
        self.ui.loadEdgeImg.clicked.connect(lambda: self.getPictures(5))
        self.ui.hybrid1.clicked.connect(self.Hybrid)
        self.ui.set.clicked.connect(self.Normalization)
        self.ui.gray.clicked.connect(lambda: self.getHistogram(self.grayImg, 'grey'))
        self.ui.color.clicked.connect(lambda: self.getHistogram(self.image, ' '))
        self.ui.cumcolor.clicked.connect(lambda: self.getHistogram(self.image, 'c'))
     
    
    def snakeAlgorithm(self):
        self.snakeFlag = True

        while self.snakeFlag:
            snakeImg = self.snake.visualize()
            # cv2.imshow( snake_window_name, snakeImg ) # display Snake
            self.display(snakeImg)
            x = [i[0] for i in self.snake.points]
            y = [j[1] for j in self.snake.points]
            area=0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
            area=np.abs(area)/10000
            self.ui.Area.setText(str(area))
            perimeter = self.snake.get_length()/100
            perimeter = round(perimeter,2)
            self.ui.Perimeter.setText(str(perimeter))
            snake_changed = self.snake.step()
            k = cv2.waitKey(33)
            if k == 27:
                self.snakeFlag = False
                break
    
    def clear(self):
        self.ui.Area.setText(str(''))
        self.ui.Perimeter.setText(str(''))
        self.display(self.image)
        self.snakeFlag = False
    
          
    def display(self,snakeImg):
        cv2.imwrite(r"./images/grayPicture.png", snakeImg)
        self.ui.InputImageTab5.setPixmap(QPixmap("./images/grayPicture.png"))


    def getPictures(self, tab):
        path, extention = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "")
        if path == "":
            pass
        else:
            self.image = cv2.imread(path)
            self.grayImg = self.rgb2gray(self.image)
        
            cv2.imwrite(r"./images/grayPicture.png", self.grayImg)
            self.padded = self.padding(self.grayImg, self.n)
            self.cannyImg =self.canny(1)
            if (tab == 1):
                self.ui.inputTab1.setPixmap(QPixmap(path))
            elif (tab == 2):
                self.ui.inputTab2.setPixmap(QPixmap(path))
                
            elif (tab == 3):
                self.ui.input1Tab3.setPixmap(QPixmap(path))
                self.LowCompImage = self.padded
            elif (tab == 4):
                self.ui.input2Tab3.setPixmap(QPixmap(path))
                self.HighCompImage = self.padded
    
            if ( tab == 5 ):
                self.ui.InputTab4.setPixmap(QPixmap(path))
           
            elif ( tab == 6 ):
                self.snake = snake.Snake(self.image)
                self.ui.InputImageTab5.setPixmap(QPixmap(path))

    def canny(self,flag):
        smoothedImg = self.gaussianFilterForCanny()
        mg, angle = self.highFilterOfCanny(smoothedImg)
        nonMaxImg = self.non_max_suppression(mg, angle)
        thresh, weak, strong = self.threshold(nonMaxImg, 0.05, 0.09)
        img_final = self.hysteresis(thresh, weak, strong)
        cv2.imwrite(r".\images\cannyEdge.png", img_final)
        if flag == 0:
            self.ui.edgeView.setPixmap(QPixmap(r".\images\cannyEdge.png"))
        if flag == 1:
            return img_final
  
    def line(self):
        image_final = self.canny(1)
        hough_imgs = self.hough_line(image_final)
        self.show_lines(self.image, hough_imgs, 0.9995)
        self.ui.edgeView.setPixmap(QPixmap(r".\images\houghLine.png"))
  
    def show_lines(self,img, hough_img, percentile):
        fig, ax = plt.subplots()

        acc, thetas, rhos = hough_img

        lines = self.extract_lines(*hough_img, percentile)

        ax.imshow(img)
        ax.autoscale(False)
        plt.axis('off')
        for (rho, theta), val in lines.items():
            a = np.cos(theta)
            b = np.sin(theta)
            width = 1000
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + width * (-b))
            y1 = int(y0 + width * (a))
            x2 = int(x0 - width * (-b))
            y2 = int(y0 - width * (a))
            ax.plot((x1, x2), (y1, y2), '-r')

        plt.savefig(r'.\images\houghLine.png', bbox_inches='tight', pad_inches=0)

    def extract_lines(self,accumulator, thetas, rhos, percentile):
        lines = defaultdict()
        threshold = np.quantile(accumulator, percentile)
        for rho_idx in range(accumulator.shape[0]):
            for theta_idx in range(accumulator.shape[1]):
                if accumulator[rho_idx, theta_idx] > threshold:
                    theta = thetas[theta_idx]
                    rho = rhos[rho_idx]
                    lines[(rho, theta)] = accumulator[rho_idx, theta_idx]

        return lines

    def hough_line(self, image):

        Ny = image.shape[0]
        Nx = image.shape[1]
        Maxdist = int(np.round(np.sqrt(Nx ** 2 + Ny ** 2)))

        thetas = np.deg2rad(np.arange(-90, 90))

        rs = np.linspace(-Maxdist, Maxdist, 2 * Maxdist)

        accumulator = np.zeros((2 * Maxdist, len(thetas)))

        for y in range(Ny):
            for x in range(Nx):
                if image[y, x] > 0:
                    for k in range(len(thetas)):
                        r = x * np.cos(thetas[k]) + y * np.sin(thetas[k])
                        accumulator[int(r) + Maxdist, k] += 1
        return accumulator, thetas, rs
   
    def circle(self):
        res = self.detectCircles(self.cannyImg, 15, 15, [10, 60])
        self.displayCircles(res)
        # self.ui.edgeView.setPixmap(QPixmap("./images/houghCircle.png"))

    def detectCircles(self,img, threshold, region, radius):
        # detectCircles takes img array,threshold of min points,region and radius range param
        M, N = img.shape

        [minR, maxR] = radius

        R = maxR - minR
        # Acc. array for the radius, X and Y
        # Also appending a padding of 2 times maxR.
        A = np.zeros((maxR, M + 2 * maxR, N + 2 * maxR))
        B = np.zeros((maxR, M + 2 * maxR, N + 2 * maxR))

        # making angles array
        angles = np.arange(0, 360) * np.pi / 180
        # Extracting all edge indices from filtered img
        edges = np.argwhere(img[:, :])
        for v in range(R):
            r = minR + v
            circleBprint = np.zeros((2 * (r + 1), 2 * (r + 1)))
            (m, n) = (r + 1, r + 1)  # the center of the blueprint
            for angle in angles:
                x = int(np.round(r * np.cos(angle)))
                y = int(np.round(r * np.sin(angle)))
                circleBprint[m + x, n + y] = 1
            const = np.argwhere(circleBprint).shape[0]
            for x, y in edges:
                # For each edge coordinates
                # Centering the blueprint circle over the edges
                # and updating the accumulator array
                X = [x - m + maxR, x + m + maxR]  # Computing the extreme X values
                Y = [y - n + maxR, y + n + maxR]  # Computing the extreme Y values
                A[r, X[0]:X[1], Y[0]:Y[1]] += circleBprint
            A[r][A[r] < threshold * const / r] = 0

        for r, x, y in np.argwhere(A):
            temp = A[r - region:r + region, x - region:x + region, y - region:y + region]
            try:
                p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
            except:
                continue
            B[r + (p - region), x + (a - region), y + (b - region)] = 1

        return B[:, maxR:-maxR, maxR:-maxR]

    def displayCircles(self,A):
        fig = plt.figure()
        plt.imshow(self.image)
        circleCoordinates = np.argwhere(A)  # Extracting the circle information
        circle = []
        for r, x, y in circleCoordinates:
            circle.append(plt.Circle((y, x), r, color=(1, 0, 0), fill=False))
            fig.add_subplot(111).add_artist(circle[-1])
        plt.axis('off')
        plt.savefig('houghCircle.png', bbox_inches='tight', pad_inches=0)
        self.ui.edgeView.setPixmap(QPixmap("houghCircle.png"))


    def rgb2gray(self, rgb_image):
        return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])

    def gaussianFilterForCanny(self):
        n = 5
        sigma = 1.4
        kernel = self.gaussian_kernel(n, sigma)
        img_smoothed = convolve(self.grayImg, kernel)
        return img_smoothed

    def gaussian_kernel(self,size, sigma):
        size = int(size) // 2
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        normal = 1 / (2.0 * np.pi * sigma ** 2)
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
        return g

    def highFilterOfCanny(self,img):
        maskX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        maskY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        n = 3
        R, C = img.shape
        s1values = np.zeros((R + n - 1, C + n - 1))
        s2values = np.zeros((R + n - 1, C + n - 1))
        newImage = np.zeros((R + n - 1, C + n - 1))
        for i in range(1, R - 2):
            for j in range(1, C - 2):
                S1 = np.sum(np.multiply(maskX, img[i:i + n, j:j + n]))
                S2 = np.sum(np.multiply(maskY, img[i:i + n, j:j + n]))
                s1values[i + 1, j + 1] = S1
                s2values[i + 1, j + 1] = S2
                newImage[i, j] = np.sqrt(np.power(S1, 2) + np.power(S2, 2))

        angles = np.arctan2(s2values, s1values)
        newImage *= 255.0 / newImage.max()
        return newImage, angles

    def non_max_suppression(self,img, D):
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img[i, j + 1]
                        r = img[i, j - 1]
                    # angle 45
                    elif 22.5 <= angle[i, j] < 67.5:
                        q = img[i + 1, j - 1]
                        r = img[i - 1, j + 1]
                    # angle 90
                    elif 67.5 <= angle[i, j] < 112.5:
                        q = img[i + 1, j]
                        r = img[i - 1, j]
                    # angle 135
                    elif 112.5 <= angle[i, j] < 157.5:
                        q = img[i - 1, j - 1]
                        r = img[i + 1, j + 1]

                    if (img[i, j] >= q) and (img[i, j] >= r):
                        Z[i, j] = img[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass
        return Z

    def threshold(self,img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
        highThreshold = img.max() * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio

        M, N = img.shape
        res = np.zeros((M, N), dtype=np.int32)

        weak = np.int32(25)
        strong = np.int32(255)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res, weak, strong

    def hysteresis(self,img, weak, strong=255):
        M, N = img.shape
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if img[i, j] == weak:
                    try:
                        if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                                or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                                or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                        img[i - 1, j + 1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass
        return img
        
    def Hybrid(self):
        imageSmoothed = self.gaussianFilter(self.LowCompImage)
        self.ui.outputTab1.setText("Output Image")
        imageSharped = self.LaplacianFilter(self.HighCompImage, 3)
        outputImage = (imageSmoothed * (1 - self.alpha)) + (imageSharped * self.alpha)
        cv2.imwrite(r"./images/HybridImage.png", outputImage)
        self.ui.outputTab3.setPixmap(QPixmap(r"./images/HybridImage.png"))

   
    def salt_pepper_noise(self, img, percent):
        img_noisy = np.zeros(img.shape)
        salt_pepper = np.random.random(img.shape)  # Uniform distribution
        cleanPixels_ind = salt_pepper > percent
        NoisePixels_ind = salt_pepper <= percent
        pepper = (salt_pepper <= (0.5 * percent))  # pepper < half percent
        salt = ((salt_pepper <= percent) & (salt_pepper > 0.5 * percent))
        img_noisy[cleanPixels_ind] = img[cleanPixels_ind]
        img_noisy[pepper] = 0
        img_noisy[salt] = 1
        cv2.imwrite(r".\images\noise.png", img_noisy)
        self.ui.outputTab1.setPixmap(QPixmap(r".\images\noise.png"))
        return (img_noisy)

    def chooseFilter(self):
        if str(self.ui.selectTab1.currentText()) == "Salt And Pepper":
            self.ui.groupBox_2.hide()
            self.noiseImage = self.salt_pepper_noise(self.padded, self.precent)
        if str(self.ui.selectTab1.currentText()) == "Average Filter":
            self.ui.groupBox_2.hide()
            self.FilterImage = self.AvgFilter(self.noiseImage, 3)
        if str(self.ui.selectTab1.currentText()) == "Gaussian Filter":
            self.ui.groupBox_2.hide()
            self.FilterImage = self.gaussianFilter(self.noiseImage)
        if str(self.ui.selectTab1.currentText()) == "Median Filter":
            self.ui.groupBox_2.hide()
            self.FilterImage = self.medianFilter(self.noiseImage, 3)
        if str(self.ui.selectTab1.currentText()) == "Sobel Filter":
            self.ui.groupBox_2.hide()
            maskX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            maskY = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
            file = r"./images/SobelFilter.png"
            self.highFilter(self.FilterImage, 3, maskX, maskY, file)
        if str(self.ui.selectTab1.currentText()) == "Roberts Filter":
            self.ui.groupBox_2.hide()
            maskX = [[1, 0], [0, -1]]
            maskY = [[0, 1], [-1, 0]]
            file = r"./images/RobertsFilter.png"
            self.highFilter(self.FilterImage, 2, maskX, maskY, file)
        if str(self.ui.selectTab1.currentText()) == "Prewitt Filter":
            self.ui.groupBox_2.hide()
            maskX = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
            maskY = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
            file = r"./images/PrewittFilter.png"
            self.highFilter(self.FilterImage, 3, maskX, maskY, file)
        if str(self.ui.selectTab1.currentText()) == "Normalization":
            self.ui.groupBox_2.show()
            # self.Normalization()
        if str(self.ui.selectTab1.currentText()) == "Equalization":
            self.ui.groupBox_2.hide()
            self.Equalization()
        if str(self.ui.selectTab1.currentText()) == "Local Thresholding":
            self.ui.groupBox_2.hide()
            self.LocalThresholding()
        if str(self.ui.selectTab1.currentText()) == "Global Thresholding":
            self.ui.groupBox_2.hide()
            self.GlobalThresholding()
        if str(self.ui.selectTab1.currentText()) == "Low Frequency Filter":
            self.ui.groupBox_2.hide()
            self.LowFreqFilter()
        if str(self.ui.selectTab1.currentText()) == "High Frequency Filter":
            self.ui.groupBox_2.hide()
            self.HighFreqFilter()

    def Normalization(self):
        img = self.grayImg
        minIntensity = np.int(self.ui.min.text())
        maxIntensity = np.int(self.ui.max.text())
        img = ((img - minIntensity) * maxIntensity) / (maxIntensity - minIntensity)
        cv2.imwrite(r".\images\normalized.png", img)
        self.ui.outputTab1.setPixmap(QPixmap(r".\images\normalized.png"))

    def Equalization(self):

        array = np.around(self.grayImg).astype(int)
        histo, bins_edges = np.histogram(array.flatten(), bins=256, range=(0, 256))
        chistogram_array = np.cumsum(histo)
        chistogram_array = chistogram_array * (255 / chistogram_array.max())
        img_list = list(array.flatten())
        eq_img_list = [chistogram_array[p] for p in img_list]
        eq_img_array = np.reshape(np.asarray(eq_img_list), array.shape)
        cv2.imwrite(r"./images/equalized.png", eq_img_array)
        self.ui.outputTab1.setPixmap(QPixmap(r"./images/equalized.png"))

    def LocalThresholding(self):
        n = 5  # mask 5 x 5
        newImg = np.zeros((self.R, self.C))
        img = self.padding(self.grayImg, n)
        R, C = img.shape
        for i in range(R - n // 2):
            for j in range(C - n // 2):
                mask = np.mean(img[i:i + n, j:j + n])
                print(mask)
                newImg[i, j] = self.maxIntensity if mask >= self.Th else self.minIntensity
        cv2.imwrite(r".\images\LocalThresholding.png", newImg)
        self.ui.outputTab1.setPixmap(QPixmap(r".\images\LocalThresholding.png"))
        # return(newImg)

    def GlobalThresholding(self):
        img = self.grayImg
        newImg = np.zeros((self.R, self.C))
        for i in range(self.R):
            for j in range(self.C):
                newImg[i, j] = self.maxIntensity if img[i, j] >= self.Th else self.minIntensity
        cv2.imwrite(r".\images\GlobalThresholding.png", newImg)
        self.ui.outputTab1.setPixmap(QPixmap(r".\images\GlobalThresholding.png"))
        # return(newImg)

    def LowFreqFilter(self):
        self.freqFilter(self.grayImg, "a")

    def HighFreqFilter(self):
        self.freqFilter(self.grayImg, "p")

    def gaussianFilter(self, img):
        n = 3
        sigma = 1
        kernel = self.gaussian_kernel(n, sigma)
        filteredImg = np.zeros((self.R, self.C))
        for i in range(self.R - n // 2):
            for j in range(self.C - n // 2):
                window = img[i: i + n, j: j + n]
                filteredImg[i, j] = np.sum(window * kernel)
        cv2.imwrite(r".\images\GaussianFilter.png", filteredImg)
        self.ui.outputTab1.setPixmap(QPixmap(r".\images\GaussianFilter.png"))
        return filteredImg

    def gaussian_kernel(self, k_size, sigma):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
        return g

    def medianFilter(self, img, n):
        R, C = img.shape
        FilteredImage = np.zeros((self.R, self.C))
        for i in range(self.R - n // 2):
            for j in range(self.C - n // 2):
                mask = img[i:i + n, j:j + n]
                FilteredImage[i, j] = np.median(mask[:])
        cv2.imwrite(r".\images\MedianFilter.png", FilteredImage)
        self.ui.outputTab1.setPixmap(QPixmap(r".\images\MedianFilter.png"))
        return (FilteredImage)

    def LaplacianFilter(self, img, n):
        R, C = img.shape
        mask = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        newImage = np.zeros((self.R, self.C))
        for i in range(self.R - n // 2):
            for j in range(self.C - n // 2):
                newImage[i + 1, j + 1] = np.sum(np.sum(np.multiply(mask, img[i:i + n, j:j + n])))
        newImage *= 255.0 / newImage.max()
        cv2.imwrite(r"./images/LaplacianFilter.png", newImage)
        return newImage

    def AvgFilter(self, img, n):
        R, C = img.shape
        n = 3
        mask = np.ones((n, n), np.float32) / (n * n)
        FilteredImage = np.zeros((self.R, self.C))
        for i in range(self.R - n // 2):
            for j in range(self.C - n // 2):
                FilteredImage[i, j] = np.sum(np.multiply(mask, img[i:i + n, j:j + n]))
        cv2.imwrite(r".\images\AvgFilter.png", FilteredImage)
        self.ui.outputTab1.setPixmap(QPixmap(r".\images\AvgFilter.png"))
        return (FilteredImage)

    def highFilter(self, img, n, maskX, maskY, file):
        R, C = img.shape
        newImage = np.zeros((R + n - 1, C + n - 1))
        newImage = np.zeros((R + n - 1, C + n - 1))
        for i in range(1, R - 2):
            for j in range(1, C - 2):
                S1 = np.sum(np.multiply(maskX, img[i:i + n, j:j + n]))
                S2 = np.sum(np.multiply(maskY, img[i:i + n, j:j + n]))
                newImage[i + 1, j + 1] = np.sqrt(np.power(S1, 2) + np.power(S2, 2))
        newImage *= 255.0 / newImage.max()
        cv2.imwrite(file, newImage)
        self.ui.outputTab1.setPixmap(QPixmap(file))

    def padding(self, img, n):
        self.R, self.C = img.shape
        imgAfterPadding = np.zeros((self.R + self.n - 1, self.C + self.n - 1))
        imgAfterPadding[1:1 + self.R, 1:1 + self.C] = img.copy()
        return (imgAfterPadding)

    def paddingGeneral(self, desiredSize, img, n, backColor):
        # desired size img
        row, col = desiredSize.shape
        if backColor == 'w':
            imgAfterPadding = np.ones((row, col))
        elif backColor == 'b':
            imgAfterPadding = np.zeros((row, col))
        # center offset
        xx = (row - n) // 2
        yy = (col - n) // 2
        imgAfterPadding[xx:xx + n, yy:yy + n] = img.copy()
        return (imgAfterPadding)

    def getHistogram(self, img, type):
        if type == 'grey':
            histo, bins_edges = np.histogram(img[:, :], bins=256, range=(0, 256))
            self.ui.inputHistogram.plot(bins_edges[0:-1], histo)
        # intenisties = np.arange(256)
        # histo = np.bincount(img[2], minlength=256)
        else:
            colors = ('r', 'g', 'b')
            self.ui.inputHistogram.clear()
            for idx, color in enumerate(colors):
                histo, bins_edges = np.histogram(img[:, :, idx], bins=256, range=(0, 256))
                self.ui.inputHistogram.setBackground('w')
                if type == 'c':
                    histo = np.cumsum(histo)
                self.ui.inputHistogram.plot(bins_edges[0:-1], histo, pen=color)

    def freqFilter(self, img, filterType):
        img_fshift = self.fourrier(img)
        # avg filter low pass fft
        if filterType == "a":
            mask = np.ones((3, 3), np.float32) / 9
            paddedMask = self.paddingGeneral(img, mask, 3, 'b')
            maskFFT = self.fourrier(paddedMask)
            resultImg = maskFFT * img_fshift
            newImg = self.inverseFourrier(resultImg)
        # laplacian high pass fft
        if filterType == "p":
            mask = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])
            paddedMask = self.paddingGeneral(img, mask, 3, 'b')
            maskFFT = self.fourrier(paddedMask)
            test = np.log(np.abs(maskFFT))
            cv2.imwrite(r"./images/maskfft.png", test)
            resultImg = maskFFT * img_fshift
            newImg = self.inverseFourrier(resultImg)

        cv2.imwrite(r"./images/fourrierTest.png", newImg)
        self.ui.outputTab1.setPixmap(QPixmap(r"./images/fourrierTest.png"))

        # self.ui.graphicsView.image(magnitude_spectrum)
        print("fft then send it to filter func")

    def fourrier(self, img):
        fourrier = np.fft.fft2(img)
        fshift = np.fft.fftshift(fourrier)
        return fshift

    def inverseFourrier(self, fourrImg):
        # img_back = np.fft.ifftshift(fourrImg)
        img_back = np.fft.ifft2(fourrImg)
        img_back = np.fft.ifftshift(img_back)
        img_back = np.abs(img_back)
        return img_back


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
