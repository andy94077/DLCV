import numpy as np
import cv2

img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)

# (1)
blur = cv2.GaussianBlur(img, (3, 3), 1 / (2 * np.log(2)))
cv2.imwrite('3-1.png', blur)

# (2)
kx = np.array([[-0.5, 0., 0.5]])
ky = np.array([[-0.5], [0], [0.5]])
img_x = cv2.filter2D(img, -1, kx)
img_y = cv2.filter2D(img, -1, ky)
cv2.imwrite('3-2_x.png', np.clip(img_x * 10, 0, 255).astype(np.uint8))
cv2.imwrite('3-2_y.png', np.clip(img_y * 10, 0, 255).astype(np.uint8))

# (3)
grad_img = np.sqrt(img_x ** 2 + img_y ** 2)
grad_blur = np.sqrt(cv2.filter2D(blur, -1, kx) ** 2 + cv2.filter2D(blur, -1, ky) ** 2)

cv2.imwrite('3-3_original.png', np.clip(grad_img * 7, 0, 255).astype(np.uint8))
cv2.imwrite('3-3_blur.png', np.clip(grad_blur * 7, 0, 255).astype(np.uint8))
