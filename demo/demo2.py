
import cv2
import numpy as np

#imgPath = 'demo5.jpeg'#5, 有明显的水印——先去水印再分割
#imgPath = 'demo12.jpeg'#8/12/13/17, 板件上有大块的白色区域——2、区域分裂合并分割
#imgPath = 'demo19.jpeg'#2/9/19, 背景色彩太过丰富
imgPath = 'demo9.jpeg'#普通图片直接用这种方法分割就可以

imgRoot = cv2.imread(imgPath)

#一、图像变换——伽马变换

#转灰度
img1 = cv2.cvtColor(imgRoot, cv2.COLOR_BGR2GRAY)

fi = img1 / 255.0
#####(b). 伽马变换
gamma = 2
img = np.power(fi, gamma)
cv2.imshow("img", img1)
cv2.imshow("out", img)
img = np.uint8(np.clip((1.5*img1 + 15), 0, 255))
#二、图像转灰度——直方图增强——然后图像平滑化

#直方图——二插法
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img = clahe.apply(img)

#平滑化——高斯滤波
gray = cv2.GaussianBlur(img, (11, 11), 0)
cv2.imshow("gray", gray)

####################################################
#三、提取图像的梯度——边缘检测-多种方法->多种方法放到一个函数里#
####################################################
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1)
# 以Sobel算子计算x，y方向上的梯度，
# 之后在x方向上减去y方向上的梯度，
# 通过这个减法，
# 我们留下具有高水平梯度和低垂直梯度
# 的图像区域。
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
#gradient = cv2.Canny(gray, 200, 300)
cv2.imshow("gradient", gradient)

#四、图像二值化：
blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
cv2.imshow("blurred", blurred)

#五、图像形态学——腐蚀膨胀：
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closed1", closed)
#图像开运算
closed = cv2.dilate(closed, None, iterations = 6)#膨胀
closed = cv2.erode(closed, None, iterations = 1)#腐蚀

cv2.imshow("closed", closed)
###########################################
#六、寻找轮廓——试试其他算法->多种方法放到一个函数里#
###########################################
# (_, cnts, _) = cv2.findContours(
#     参数一： 二值化图像
#     closed.copy(),
#     参数二：轮廓类型
#     # cv2.RETR_EXTERNAL, #表示只检测外轮廓
#     # cv2.RETR_CCOMP, #建立两个等级的轮廓,上一层是边界
#     # cv2.RETR_LIST, #检测的轮廓不建立等级关系
#     # cv2.RETR_TREE, #建立一个等级树结构的轮廓
#     # cv2.CHAIN_APPROX_NONE, #存储所有的轮廓点，相邻的两个点的像素位置差不超过1
#     参数三：处理近似方法
#     # cv2.CHAIN_APPROX_SIMPLE, #例如一个矩形轮廓只需4个点来保存轮廓信息
#     # cv2.CHAIN_APPROX_TC89_L1,
#     # cv2.CHAIN_APPROX_TC89_KCOS
#     )
(_, cnts, _) = cv2.findContours(
        closed.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
rect = cv2.minAreaRect(c)

box = np.int0(cv2.boxPoints(rect))
draw_img = cv2.drawContours(imgRoot.copy(), [box], -1, (0,255,255), 3)
cv2.imshow("draw_img", draw_img)
cv2.imwrite('draw_img.jpeg', draw_img)

cv2.waitKey()
cv2.destroyAllWindows()
#七、 切割图像：#多种切割方法——写成一个函数
pass