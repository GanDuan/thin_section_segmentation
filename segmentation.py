%matplotlib qt5
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
from skimage import measure
from skimage import restoration
from skimage import img_as_float
import pandas as pd
from skimage import segmentation


#import image
filename = "yourfilename"
im = cv2.imread(f'{filename}.jpg', 0)
plt.imshow(im, cmap="gray")

#Denoise image
im_float = img_as_float(im)
sigma = restoration.estimate_sigma(im_float)
im_denoised = restoration.denoise_nl_means(im_float, h=sigma)
plt.imshow(im_denoised, cmap='gray')
ax = plt.axis('off')

#User defined markers
#First time, need to click in the center in each grain recgonized by eyes
plt.imshow(im_denoised, cmap='gray')
click_markers = plt.ginput(n=-1, timeout=-1)
arr = np.asarray(click_markers)
pd.DataFrame(arr).to_csv(f"{filename}_markers_coords.csv")

##if run the script for a few times, then can load the previously stored coordinates
df = pd.read_csv(f"{filename}_markers_coords.csv")
arr = df.to_numpy().T
x = arr[1]
y = arr[2]

plt.imshow(im_denoised.data, cmap='gray')
plt.plot(x, y, 'or', ms=2)

markers = np.zeros(im_denoised.data.shape, dtype=int)
markers[y.astype(int), x.astype(int)] = np.arange(len(x)) + 1
markers = morphology.dilation(markers, morphology.disk(5))

#Watershed segmentation
# Black tophat transformation (see https://en.wikipedia.org/wiki/Top-hat_transform)
hat = ndimage.black_tophat(im_denoised, 7)
# Combine with denoised image
hat -= 0.3 * im_denoised
# Morphological dilation to try to remove some holes in hat image
hat = morphology.dilation(hat)

labels_hat = segmentation.watershed(hat, markers, watershed_line=True)
plt.imshow(labels_hat, cmap="cool")

#Measure grains
clusters = measure.regionprops(labels_hat, im_denoised)
propList = ['Area',
            'equivalent_diameter', #Added... verify if it works
            'orientation', #Added, verify if it works. Angle btwn x-axis and major axis.
            'MajorAxisLength',
            'MinorAxisLength',
            'Perimeter',
            'MinIntensity',
            'MeanIntensity',
            'MaxIntensity', 
            ] 
pixels_to_um = 26 #change the value accordingly
output_file = open(f'tophat_watershed_{filename}.csv', 'w')
output_file.write(',' + ",".join(propList) + '\n') #join strings in array by commas, leave first cell blank
#First cell blank to leave room for header (column names)

for cluster_props in clusters:
    #output cluster properties to the excel file
    output_file.write(str(cluster_props['Label']))
    for i,prop in enumerate(propList):
        if(prop == 'Area'): 
            to_print = cluster_props[prop]*pixels_to_um**2   #Convert pixel square to um square
        elif(prop == 'orientation'): 
            to_print = cluster_props[prop]*57.2958  #Convert to degrees from radians
        elif(prop.find('Intensity') < 0):          # Any prop without Intensity in its name
            to_print = cluster_props[prop]*pixels_to_um
        else: 
            to_print = cluster_props[prop]     #Reamining props, basically the ones with Intensity in its name
        output_file.write(',' + str(to_print))
    output_file.write('\n')
output_file.close()   #Closes the file, otherwise it would be read only. 
