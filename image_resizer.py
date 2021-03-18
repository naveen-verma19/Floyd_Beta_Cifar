from skimage import data, color
from skimage.transform import rescale, resize
from skimage.color import rgb2gray,gray2rgb,rgb2lab
import os

from skimage import io
path ="/Users/naveen/Downloads/img_align_celeba/"
outpath="/Users/naveen/Downloads/Celeb 128x128/"
# files_sorted=os.listdir(path)
# files_sorted = [x for x in files_sorted if x.endswith(".jpg")]
# files_sorted.sort(key=lambda x:int(x.split(".")[0]))
#
# X = []
# for filename in files_sorted[6000:8000]:
#     image= io.imread(path+filename)
#     image_resized = resize(image, (128, 128),
#                            anti_aliasing=True)
#     io.imsave(outpath+filename,image_resized)


image =io.imread("chris.jpg")
image_resized = resize(image, (128, 128),anti_aliasing=True)
grayscale = rgb2gray(image_resized)
rg= gray2rgb(grayscale)
io.imsave("testgray2.jpg",rg)

print("Done!")