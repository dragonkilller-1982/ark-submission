import cv2        #ironman
import numpy as np

img = cv2.imdecode(np.fromfile(r"C:\Users\soumy\Downloads\iron_man_noisy.jpg", np.uint8), 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
clean = np.zeros_like(img)

for c in contours:
    if cv2.contourArea(c) >= 10 or max(cv2.boundingRect(c)[2:]) >= 6:
        mask = cv2.dilate(cv2.drawContours(np.zeros_like(gray), [c], -1, 255, -1),
                          np.ones((2, 2), np.uint8))
        clean[mask > 0] = img[mask > 0]

clean = cv2.fastNlMeansDenoisingColored(clean, None, 3, 3, 7, 21)

cv2.imencode(".jpg", clean)[1].tofile(r"C:\Users\soumy\Downloads\iron_man_clean.jpg")
print("Done! Saved iron_man_clean.jpg")

h, w = img.shape[:2]
s = (int(w * 0.4), int(h * 0.4))
cv2.imshow("Before | After", np.hstack((cv2.resize(img, s), cv2.resize(clean, s))))
cv2.waitKey(0)
cv2.destroyAllWindows()