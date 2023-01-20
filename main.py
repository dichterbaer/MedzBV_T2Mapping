import numpy as np
import os
import SimpleITK as sitk
from matplotlib import pyplot as plt
import scipy

imagepath = r'C:\Users\MStempel\Desktop\LWS\Referenzdaten_Siemens_Proband02_LWS_T2mapping+Volumetry\\'


def t2exp(x, t, t2):
    return np.exp(-(t/t2))


def list_files(dir_path):
    files = os.listdir(dir_path)
    return [file for file in files if file.startswith("REF02_1.MR.2")]

files = list_files(imagepath)
images = []

for i, file in enumerate(files[6:]):
    img = sitk.ReadImage(imagepath + file)
    slice_loc = float(img.GetMetaData('0020|1041'))
    if (slice_loc < -2 and slice_loc > -3): 
        images.append(img)

px_values = []
echotimes = []
pos = [128, 145]
for img in images:
    im = sitk.GetArrayFromImage(img)
    im = np.squeeze(im)
    px_values.append(im[145, 128])
    # get echo time
    echotimes.append(float(img.GetMetaData('0018|0081')))

echotimes = np.array(echotimes)
#images_np = np.array(images)
px_vals = np.array(px_values)
sorted_indices = np.argsort(echotimes)
echotimes_sorted = echotimes[sorted_indices]
#images_sorted = images[sorted_indices]
px_values_sorted = px_vals[sorted_indices]

t2_start = 250
#t2_fit, cv = scipy.optimize.curve_fit(t2exp, echotimes_sorted, px_values_sorted, t2_start)
t2 = np.polyfit(np.log(echotimes_sorted), px_values_sorted, 1)

plt.plot(np.log(echotimes_sorted), px_values_sorted, '*-')
plt.show()

