import numpy as np
import os
import SimpleITK as sitk
from matplotlib import pyplot as plt
import cv2

imagepath = r'LWS\Referenzdaten_Siemens_Proband02_LWS_T2mapping+Volumetry\\'


def define_joint_regions():
    '''Define joint regions by drawing contours on the image and save the contours as a mask'''
    image = cv2.imread('joint_mask.png', 0)
    _, thresh = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    masks = []
    for contour in contours:
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        masks.append(mask)

    # Sort masks by their y position
    masks = sorted(masks, key=lambda x: np.min(np.where(x == 255)[0]))

    #show masks for debugging
    if(False):
        for mask in masks:
            plt.imshow(mask, cmap='gray')
            plt.show()
    return masks

def get_image_data(imagepath):
    files = list_files(imagepath)
    images_mr = []

    for i, file in enumerate(files[6:]):
        img = sitk.ReadImage(imagepath + file)
        slice_loc = float(img.GetMetaData('0020|1041'))
        if (slice_loc < -2 and slice_loc > -3): 
            images_mr.append(img)

    px_values = []
    echotimes = []
    images_np = []
    pos = [128, 145]
    for img in images_mr:
        im = sitk.GetArrayFromImage(img)
        im = np.squeeze(im)
        images_np.append(im)
        # get echo time
        echotimes.append(float(img.GetMetaData('0018|0081')))
    
    # sort images by echo time
    echotimes = np.array(echotimes)
    sorted_indices = np.argsort(echotimes)
    echotimes_sorted = echotimes[sorted_indices]
    images_np = np.array(images_np)
    images_np_sorted = images_np[sorted_indices]
    images_mr_sorted = np.array(images_mr)[sorted_indices]
    return images_mr_sorted, images_np_sorted, echotimes_sorted


def t2exp(x, t, t2):
    return np.exp(-(t/t2))


def list_files(dir_path):
    files = os.listdir(dir_path)
    return [file for file in files if file.startswith("REF02_1.MR.2")]

masks = define_joint_regions()
images_mr, images_np, echotimes = get_image_data(imagepath)

#iterate over all masks and calculate the t2 value for each pixel in the mask and write them into a new img
t2_images = []
t2_means = []
plt.figure(figsize=(20, 20))
ncols = 3 
nrows = (len(masks)+1) // ncols + ((len(masks)+1) % ncols > 0)
for i, mask in enumerate(masks):
    t2_image = np.zeros_like(images_np[i])
    t2s = []
    #initialize plot with number of subplots equal to the number of masks
    #iterate over each pixel in the mask
    ax = plt.subplot(nrows, ncols, i + 1)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if(mask[x,y] == 255):
                px_values = []
                #iterate over all images and get the pixel values for the mask
                for img in images_np:
                    px_values.append(img[x,y])
                px_values = np.array(px_values)
                #t2_start = 0.1
                #t2_fit, cv = scipy.optimize.curve_fit(t2exp, echotimes, px_values, t2_start)
                t2 = np.polyfit(-np.log(echotimes), px_values, 1)
                t2_image[x,y] = t2[0]
                #plot t2 values for each pixel
                ax.plot(np.log(echotimes), px_values, 'o--')
                t2s.append(t2[0])   
    title = 'T2 values for joint region ' + str(i+1) + ' (mean: ' + "{:.2f}".format(np.mean(t2s)) + 'ms)'
    ax.set_title(title)
    ax.set_xlabel('Echo time [ms]')
    ax.set_ylabel('Signal intensity')

    t2_images.append(t2_image)
    t2_means.append(np.mean(t2s))
    #print progress
    print(str(i+1) + '/' + str(len(masks)))
#show all plots

#combine all t2 images into one image
t2_image = np.zeros_like(t2_images[0])
for img in t2_images:
    t2_image += img
ax = plt.subplot(nrows, ncols, len(masks)+1)
ax.imshow(t2_image)
plt.show()


plt.imshow(t2_image)
plt.show()

print(t2_means)


    
# #iterate over all images and get the pixel values for the mask
# for img in images_np:
#     px_values.append(img[np.where(mask == 255)])
# px_values = np.array(px_values)
# px_values_sorted = px_values[np.argsort(echotimes)]
# #t2_start = 0.1
# #t2_fit, cv = scipy.optimize.curve_fit(t2exp, echotimes, px_values, t2_start)
# t2 = np.polyfit(np.log(echotimes), px_values, 1)
# print(t2)
# #plt.plot(np.log(echotimes), px_values, '*-')
# #plt.show()
# #print()

# for i, img in enumerate(images_np_sorted):
#     plt.imshow(img, cmap='gray')
#     plt.show()
#     #save each image as png with the echo time as filename
#     plt.imsave(str(echotimes_sorted[i]) + '.png', img, cmap='gray')


# #t2_fit, cv = scipy.optimize.curve_fit(t2exp, echotimes_sorted, px_values_sorted, t2_start)
# t2 = np.polyfit(np.log(echotimes_sorted), px_values_sorted, 1)

# plt.plot(np.log(echotimes_sorted), px_values_sorted, '*-')
# plt.show()
# print()

