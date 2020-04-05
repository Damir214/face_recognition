import cv2
import os

from matplotlib import pyplot as plt


class ViolaDetector:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml',
    )

    line_width = 3
    face_color = (0, 0, 255)
    eyes_color = (0, 255, 0)
    scale_factor = 1.1
    min_neighbors = 6

    def matcher(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, self.scale_factor, self.min_neighbors,
        )
        for x, y, w, h in faces:
            img = cv2.rectangle(
                img, (x, y), (x + w, y + h), self.face_color, self.line_width,
            )
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = img[y : y + h, x : x + w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(
                    roi_color,
                    (ex, ey),
                    (ex + ew, ey + eh),
                    self.eyes_color,
                    self.line_width,
                )

        plt.imshow(img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        plt.show()


if __name__ == '__main__':
    file_directory = 'data/sample'
    images_count = 8
    images = [
        os.path.join(file_directory, f'{i}.jpg')
        for i in range(1, images_count + 1)
    ]
    viola_detector = ViolaDetector()
    for image in images:
        print(f'detecting {image}')
        viola_detector.matcher(image)

    file_directory = 'data/another_sample'
    images_count = 4
    images = [
        os.path.join(file_directory, f'{i}.png')
        for i in range(1, images_count + 1)
    ]

    for image in images:
        print(f'detecting {image}')
        viola_detector.matcher(image)
