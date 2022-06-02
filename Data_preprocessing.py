import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import glob
from numpy import inf
np.seterr(divide='ignore', invalid='ignore')

## importing multiple volumes
folders = glob.glob('/home/student2/Desktop/Working_Kunal/IXI-MRA/Guys/*')
imagenames_list_MRA = []
imagenames_list_T1 = []
for folder in folders:
    for f in glob.glob(folder+'/*-MRA.nii.gz'):
        imagenames_list_MRA.append(f)
    for f in glob.glob(folder+'/*-T1toMRA.nii.gz'):
        imagenames_list_T1.append(f)

#print(imagenames_list_MRA)
"""
read_images = []

for image in imagenames_list_MRA:
    #read_images.append(nib.load(image))
    print(image)

print(imagenames_list_MRA[0])

img=nib.load(imagenames_list_MRA[0])
img_data = img.get_fdata()
"""
results_ = []

for kk in range(0,len(imagenames_list_MRA)-1):           ## initiating the loop
    img = nib.load(imagenames_list_MRA[kk])              ## load single MRA volume
    img_data = np.double(img.get_fdata())                ## convert MRA volume into data
    img_t1 = nib.load(imagenames_list_T1[kk])            ## load single T1 volume
    img_data_t1 = np.double(img_t1.get_fdata())          ## convert T1 volume into data
    div_MRA_T1 = (img_data/img_data_t1)                  ## divide MRA by T1
    div_MRA_T1_nan = div_MRA_T1[~np.isnan(div_MRA_T1)]   ## remove Nan values
    div_MRA_T1_nan[div_MRA_T1_nan >= inf] = 0            ## remove inf values
    mean_MRA_T1 = np.mean(div_MRA_T1_nan)                ## take mean of all voxels
    print(mean_MRA_T1, imagenames_list_MRA[kk])          ## print mean value and file name
    results_.extend([mean_MRA_T1, imagenames_list_MRA[kk]])                         ## store result into an array

plt.figure(1)
plt.plot(results_)                                       ## plot results
plt.show()
print(results_)

## export the results_ array into text file
np.savetxt('T1_division', results_)

############################################
############################################
#############################################

from numpy import inf
import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import glob

## volume import IXI HH T2toMRA
folders = glob.glob("/home/student2/Desktop/Working_Kunal/IXI-MRA/HH/*")
imagenames_list_T2toMRA = []
for folder in folders:
    for f in glob.glob(folder+'/*-T2toMRA.nii.gz'):
        imagenames_list_T2toMRA.append(f)

## image shape
#img_shape = img.shape
#print(img.shape)


#print(imagenames_list_T2toMRA[0])
tmp_ = imagenames_list_T2toMRA[0]
#print(tmp_[48:57])
#print(tmp_[59:87])


## convert the volume into data

for i in range(0, len(imagenames_list_T2toMRA)-1):                # initializing the loop
    img_T2toMRA = nib.load(imagenames_list_T2toMRA[i])            # loading the volume
    img_T2toMRA_data = np.double(img_T2toMRA.get_fdata())         # convert volume into data
    img_T2toMRA_shape = img_T2toMRA_data.shape                    # find and store the size of volume
    # saving the patches in variables

    p1 = img_T2toMRA_data[0:np.int(img_T2toMRA_shape[0] / 2), 0:np.int(img_T2toMRA_shape[1] / 2), :]
    #p2 = img_T2toMRA_data[0:np.int(img_T2toMRA_shape[0] / 2), np.int(img_T2toMRA_shape[1] / 2):, :]
    #p3 = img_T2toMRA_data[np.int(img_T2toMRA_shape[0] / 2):, 0:np.int(img_T2toMRA_shape[1] / 2), :]
    #p4 = img_T2toMRA_data[np.int(img_T2toMRA_shape[0] / 2):, np.int(img_T2toMRA_shape[1] / 2):, :]

    # saving the patched volume as nifti files
    p1 = nib.Nifti1Image(p1, np.eye(4))
    tmp_= imagenames_list_T2toMRA[i]
    dst_= (tmp_[0:len(tmp_)-29])
    filename_p1=tmp_[len(tmp_)-29:len(tmp_)-7]+str("-p1.nii.gz")
    print(str(dst_)+str(filename_p1))
    nib.save(p1, str(dst_)+str(filename_p1))
    #nib.save(p1, os.path.join('/home/student2/Desktop/Working_Kunal/IXI-MRA/HH/'+tmp_[48:57]))

"""
    p2 = nib.Nifti1Image(p2, np.eye(4))
    nib.save(p2, os.path.join('/home/student2/Desktop/Working_Kunal/IXI-MRA/HH/*-T2toMRA-patch-2.nii.gz'))

    p3 = nib.Nifti1Image(p3, np.eye(4))
    nib.save(p3, os.path.join('/home/student2/Desktop/Working_Kunal/IXI-MRA/HH/*-T2toMRA-patch-3.nii.gz'))

    p4 = nib.Nifti1Image(p4, np.eye(4))
    nib.save(p4, os.path.join('/home/student2/Desktop/Working_Kunal/IXI-MRA/HH/*-T2toMRA-patch-4.nii.gz'))
"""
"""
## plotting images
#plt.figure(1)
#plt.imshow(img_data[:,:,50])
#plt.show()

## plotting patches
plt.figure(2)
plt.subplot(221)
plt.imshow(img_data[0:np.int(img_shape[0]/2),0:np.int(img_shape[1]/2),50])
plt.subplot(222)
plt.imshow(img_data[0:np.int(img_shape[0]/2),np.int(img_shape[1]/2):,50])
plt.subplot(223)
plt.imshow(img_data[np.int(img_shape[0]/2):,0:np.int(img_shape[1]/2),50])
plt.subplot(224)
plt.imshow(img_data[np.int(img_shape[0]/2):,np.int(img_shape[1]/2):,50])
plt.show()


# saving the image as nifti files
p1 = nib.Nifti1Image(p1, np.eye(4))
nib.save(p1, os.path.join('/home/student2/Desktop/Working_Kunal/IXI-MRA/HH/IXI012-HH/IXI012-HH-1211-T2toMRA-patch-1.nii.gz'))

p2 = nib.Nifti1Image(p2, np.eye(4))
nib.save(p2, os.path.join('/home/student2/Desktop/Working_Kunal/IXI-MRA/HH/IXI012-HH/IXI012-HH-1211-T2toMRA-patch-2.nii.gz'))

p3 = nib.Nifti1Image(p3, np.eye(4))
nib.save(p3, os.path.join('/home/student2/Desktop/Working_Kunal/IXI-MRA/HH/IXI012-HH/IXI012-HH-1211-T2toMRA-patch-3.nii.gz'))

p4 = nib.Nifti1Image(p4, np.eye(4))
nib.save(p4, os.path.join('/home/student2/Desktop/Working_Kunal/IXI-MRA/HH/IXI012-HH/IXI012-HH-1211-T2toMRA-patch-4.nii.gz'))
"""


###############################################
###############################################
###############################################

### read text file and plot its curve
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os

txt = '/home/student2/Desktop/Working_Kunal/IXI-MRA/MRA_div_T1.txt'

# initializing empty array
results = []

## loading the complete txt file with separate lines
with open(txt) as inputfile:
    for line in inputfile:
        results.append(line.strip().split(' '))

## extracting the number from the list
#print(results[0])
variable_2 =[]
address = []
val=np.zeros(len(results))
for kk in range(0,len(results)-1):
    val[kk]=results[kk][0]
    if (val[kk]>10):
        #print(results[kk][1])
        address.append(results[kk][1])
        variable_2.append(val[kk])
"""
## plotting the numbers
plt.figure(1)
plt.plot(val)
plt.show()

plt.figure(2)
plt.plot(variable_2)
plt.show()
"""
print(address)
print(len(address))

np.savetxt('defected files.txt', address, fmt='%s')
array = []
for kk in range(0, len(address)-1):
    img = nib.load(address[kk])
    img_data = img.get_fdata()
    img_slice = img_data[:,:,50]
    array.append(img_slice)
    result_ = np.dstack(array)

result_ = nib.Nifti1Image(result_, np.eye(4))
nib.save(result_, '/home/student2/Desktop/Working_Kunal/IXI-MRA/new_img.nii.gz')


###############################################
###############################################
###############################################

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import glob
import cv2
from sklearn.preprocessing import normalize
np.seterr(divide='ignore', invalid='ignore')
from numpy import inf

## import TOF and T1 volumes
folders_1 = glob.glob("/home/student2/Desktop/Working_Kunal/IXI-MRA/Guys/*")
imagenames_list_T1 = []
for folder in folders_1:
    for f in glob.glob(folder+'/*-T1toMRA.nii.gz'):
        imagenames_list_T1.append(f)

folders_2 = glob.glob("/home/student2/Desktop/Working_Kunal/IXI-MRA/Guys/*")
imagenames_list_TOF = []
for folder in folders_2:
    for f in glob.glob(folder+'/*-MRA.nii.gz'):
        imagenames_list_TOF.append(f)

## convert volume into data and normalization for T1
#array_T1 = np.zeros([512, 512])
#print(array_T1.shape)
#array_TOF = [] #np.zeros([512, 512])
empty_stack = np.zeros((512, 1024))

for i in range(0,len(imagenames_list_T1)):

    # load nifti
    img_T1 = nib.load(imagenames_list_T1[i])
    img_TOF = nib.load(imagenames_list_TOF[i])

    # get data
    img_T1_data = np.double(img_T1.get_fdata())
    img_TOF_data = np.double(img_TOF.get_fdata())

    # extract 1 slice
    img_T1_slice = (img_T1_data[:, :, 50])
    img_TOF_slice = (img_TOF_data[:, :, 50])
    print(img_T1_slice.max(), img_TOF_slice.max())

    # normalize between 0 and 1
    xmax, xmin = img_T1_slice.max(), img_T1_slice.min()
    # x = (x – xmin) / (xmax – xmin)
    img_T1_slice = (img_T1_slice - xmin) / (xmax - xmin)
    img_T1_slice[np.isnan(img_T1_slice)]=0
    #img_T1_slice = img_T1_slice[:, ~np.isnan(img_T1_slice).any(axis=0)]     ## remove Nan values
    #img_T1_slice[img_T1_slice >= inf] = 0                    ## remove inf values

    ymax, ymin = img_TOF_slice.max(), img_TOF_slice.min()
    img_TOF_slice = (img_TOF_slice - ymin) / (ymax - ymin)
    img_TOF_slice[np.isnan(img_TOF_slice)] = 0
    #img_TOF_slice = img_TOF_slice[:, ~np.isnan(img_TOF_slice).any(axis=0)]  ## remove Nan values
    #img_TOF_slice[img_TOF_slice >= inf] = 0                  ## remove inf values

    print(img_T1_slice.max(), img_TOF_slice.max())

    conc = np.concatenate((img_TOF_slice, img_T1_slice), axis=1)
    print(conc.shape)

    empty_stack=np.dstack((empty_stack, conc))
    print(empty_stack.shape)


plt.figure(2)
plt.imshow(conc)
plt.show()

empty_stack=empty_stack[:,:,1:]
tmp_ = nib.Nifti1Image(empty_stack, np.eye(4))
nib.save(tmp_, '/home/student2/Desktop/Working_Kunal/IXI-MRA/tmp_.nii.gz')


"""
for i in range(0, len(imagenames_list_T1)-1):                # initializing the loop
    img_T1 = nib.load(imagenames_list_T1[i])                 # loading the volume
    img_T1_data = np.double(img_T1.get_fdata())              # convert volume into data
    img_T1_slice = img_T1_data[:, :, 50]                     # extract middle slice
    T1_mean = np.mean(img_T1_slice)
    T1_std = np.std(img_T1_slice)
    T1_norm = (img_T1_slice-T1_mean)/T1_std
    #T1_norm = np.all(((img_T1_slice[:]) - (min(img_T1_slice[:])))*(256/((max(img_T1_slice[:]))-(min(img_T1_slice[:]))))+0)
    #T1_norm = cv2.normalize(img_T1_slice, None, alpha=0, beta=65535)   # normalization
    array_T1.append(T1_norm)                                 # append the slice into the array
    #stack_T1 = np.dstack(array_T1)                           # stack the slices into 3D volume

## convert volume into data and normalization for TOF


for i in range(0, len(imagenames_list_TOF)-1):               # initializing the loop
    img_TOF = nib.load(imagenames_list_TOF[i])               # loading the volume
    img_TOF_data = np.double(img_TOF.get_fdata())            # convert volume into data
    img_TOF_slice = img_TOF_data[:, :, 50]                   # extract middle slice
    TOF_mean = np.mean(img_TOF_slice)
    TOF_std = np.std(img_TOF_slice)
    TOF_norm = (img_TOF_slice - TOF_mean) / TOF_std
    #TOF_norm = np.all(((img_TOF_slice[:]) - (min(img_TOF_slice[:])))*(256/((max(img_TOF_slice[:]))-(min(img_TOF_slice[:]))))+0)
    #TOF_norm = cv2.normalize(img_TOF_slice, None, alpha=0, beta=65535) # normalization
    array_TOF.append(TOF_norm)                               # append the slice into the array
    #stack_TOF = np.dstack(array_TOF)                         # stack the slices into 3D volume

## concatenate images
conc = np.concatenate((array_TOF, array_T1))
stack = np.dstack(conc)

conc = nib.Nifti1Image(conc, np.eye(4))
nib.save(conc, '/home/student2/Desktop/Working_Kunal/IXI-MRA/conc.nii.gz')

## save nifti image TOF
stack_TOF = nib.Nifti1Image(stack_TOF, np.eye(4))
nib.save(stack_TOF, '/home/student2/Desktop/Working_Kunal/IXI-MRA/stack_TOF.nii.gz')

## save nifti image T1
stack_T1 = nib.Nifti1Image(stack_T1, np.eye(4))
nib.save(stack_T1, '/home/student2/Desktop/Working_Kunal/IXI-MRA/stack_T1.nii.gz')
"""

##############################################
###############################################
##############################################
