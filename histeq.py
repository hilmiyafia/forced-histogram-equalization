import cv2
import numpy
import matplotlib.pyplot as pyplot

def HistogramEqualization(image):
    pixels = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            pixels.append((image[y, x], x, y))
    numpy.random.shuffle(pixels)
    pixels.sort(key=lambda x: x[0])
    for i, (_, x, y) in enumerate(pixels):
        image[y, x] = int(i * 256 / len(pixels))
    return image

if __name__ == "__main__":
    before = cv2.cvtColor(cv2.imread("Image.tif"), cv2.COLOR_BGR2GRAY)
    after = HistogramEqualization(numpy.copy(before))
    before_histogram, edges = numpy.histogram(before, bins=256, range=(0, 255))
    after_histogram, edges = numpy.histogram(after, bins=256, range=(0, 255))
    
    pyplot.subplot(2, 2, 1)
    pyplot.imshow(cv2.cvtColor(before, cv2.COLOR_GRAY2RGB))
    pyplot.axis("off")
    pyplot.title("Before")
    
    pyplot.subplot(2, 2, 3)
    pyplot.plot(edges[0:-1], before_histogram)
    
    pyplot.subplot(2, 2, 2)
    pyplot.imshow(cv2.cvtColor(after, cv2.COLOR_GRAY2RGB))
    pyplot.axis("off")
    pyplot.title("After")
    
    pyplot.subplot(2, 2, 4)
    pyplot.plot(edges[0:-1], after_histogram)
    pyplot.show()
