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


# image =io.imread("chris.jpg")
# image_resized = resize(image, (128, 128),anti_aliasing=True)
# grayscale = rgb2gray(image_resized)
# rg= gray2rgb(grayscale)
# io.imsave("testgray2.jpg",rg)
# im1=rgb2lab(image)
# image2 = io.imread("gray.jpg")
#
# im2=rgb2lab(image2)
# print("Done!")


path_prefix ="/Users/naveen/Documents/ML local Data/"
fnames=open("/Users/naveen/Documents/tensor_test2/places_top10_local","r").read().split("\n")
fnames=[path_prefix+x.split(" ")[0] for x in fnames]
i=0
for folder in fnames:
    i+=1
    files_sorted = os.listdir(folder)
    files_sorted = [x for x in files_sorted if x.endswith(".jpg")]
    files_sorted.sort(key=lambda x: int(x.split(".")[0]))
    for filename in files_sorted:
        image = io.imread(folder+filename)
        if image.shape[0]==128 and image.shape[1]==128:
            print("skip")
            continue
        image_resized = resize(image, (128, 128), anti_aliasing=True)
        print(i)
        io.imsave(folder+filename, image_resized)
