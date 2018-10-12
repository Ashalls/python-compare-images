# USAGE
# python compare.py

# import the necessary packages
from skimage import measure
# from skimage.measure import structural_similarity as ssim
# from matplotlib import pyplotas as plt
import matplotlib.pyplot as plt
import numpy as np
import cv2


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = measure.compare_ssim(imageA, imageB)

	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")

	# show the images
	plt.show()

# load the images -- the original, the original + contrast,
# and the original + photoshop
i7_comp = cv2.imread("images/i7-comp.png")
i7 = cv2.imread("images/i7.png")
i5 = cv2.imread("images/i5.png")
i5_comp = cv2.imread("images/i5_comp.png")
i7_2ndGen = cv2.imread("images/i7-2ndGen.png")
i5_2ndGen = cv2.imread("images/i5-2ndGen.png")

original = cv2.imread("images/jp_gates_original.png")
contrast = cv2.imread("images/jp_gates_contrast.png")
shopped = cv2.imread("images/jp_gates_photoshopped.png")

# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)
i7_comp = cv2.cvtColor(i7_comp, cv2.COLOR_BGR2GRAY)
i7 =cv2.cvtColor(i7, cv2.COLOR_BGR2GRAY)
i5 =cv2.cvtColor(i5, cv2.COLOR_BGR2GRAY)

i5_comp = cv2.cvtColor(i5_comp, cv2.COLOR_BGR2GRAY)
i7_2ndGen =cv2.cvtColor(i7_2ndGen, cv2.COLOR_BGR2GRAY)
i5_2ndGen =cv2.cvtColor(i5_2ndGen, cv2.COLOR_BGR2GRAY)

images = {
	'i7 2nd Gen' : i7_2ndGen,
	'i5 2nd Gen' : i5_2ndGen,
	'i7 7th Gen' : i7,
	'i5 7th Gen' : i5
}

def compare_function(image):
	values = {}
		# compute the mean squared error and structural similarity
	# index for the images
	for key, value in images.items():
		m = mse(image, value)
		s = measure.compare_ssim(value, image)
		# print(m)

		values[float(m)] = key
	
	print(values)

	match = min(values.keys())
	match = values.get(match)

	return match


# initialize the figure
# fig = plt.figure("Images")
# images = ("Original", original),("Contrast", contrast), ("Photoshopped", shopped)

# loop over the images
# for (i, (name, image)) in enumerate(images):
# 	# show the image
# 	ax = fig.add_subplot(1, 3, i + 1)
# 	ax.set_title(name)
# 	plt.imshow(image, cmap = plt.cm.gray)
# 	plt.axis("off")

# show the figure
# plt.show()

# compare the images
print(compare_function(i7_comp))
print(compare_function(i5_comp))


# compare_images(i7_comp, i5_2ndGen, "i7 Image vs. i7 Photo")
# compare_images(i7_comp, i7, "i7 Image vs. i7 Photo")
# # compare_images(i5, i7, "i5 image vs. i5 image")
# compare_images(i5, i7_comp, "i5 image vs. i7 photo")
# compare_images(i5_comp, i7, "i5 photo vs. i7 image")
# compare_images(i5_comp, i5, "i5 photo vs. i5 image")
# compare_images(i7_comp, i5_2ndGen, "i5 photo vs. i5 image")
# compare_images(i5_comp, i7_2ndGen, "i5 photo vs. i7 image")

# compare_images(original, contrast, "Original vs. Contrast")
# compare_images(original, shopped, "Original vs. Photoshopped")
