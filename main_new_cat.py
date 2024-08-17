import cv2
img_path=r'cat.png'
img = cv2.imread(img_path)
print(cv2.imshow('img', img))

#img=img[0:320, 150:450]
#print(cv2.imshow('new_img', img))

#print(img.shape)

## ROI (Region of Interest)
rect_img= cv2.rectangle(img, (164, 36), (480, 345), (0, 255, 0), 3)
print(cv2.imshow('img', rect_img))

roi=rect_img[36:345, 164: 480]
print(cv2.imshow('imgg', roi))

#rows, cols, _ = img.shape
#print("rows", rows)
#print("cols", cols)



cv2.waitKey(0)
cv2.destroyAllWindows()

