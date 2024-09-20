import nibabel as nib
import os
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from spinalcordtoolbox.image import Image, add_suffix
from scipy.interpolate import InterpolatedUnivariateSpline
from spinalcordtoolbox.registration.landmarks import getRigidTransformFromLandmarks
from spinalcordtoolbox.centerline.core import get_centerline 
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.scripts import sct_apply_transfo
from spinalcordtoolbox.scripts import sct_register_multimodal

os.chdir('/home/kenweber/MuscleMap/scripts/templates/abdomen')

#Load, resample, and plot images
template_filename = 'template_rpi.nii.gz'
template = Image(template_filename).change_orientation("RPI")
plt.imshow(template.data[:,:,template.data.shape[2]//2])
plt.show()

template_seg_filename = 'template_label_1_seg_rpi.nii.gz'
template_seg = Image(template_seg_filename).change_orientation("RPI")
plt.imshow(template_seg.data[:,:,template_seg.data.shape[2]//2])
plt.show()

image_filename = 'sub-test_fat.nii.gz'
image = Image(image_filename).change_orientation("RPI")
image = resample_nib(image, new_size=[0.5,0.5,0.5], new_size_type='mm', interpolation='linear', mode='nearest')
image_filename = add_suffix(image_filename, '_resampled')
image.save(image_filename, mutable=True)
plt.imshow(image.data[:,:,image.data.shape[2]//2])
plt.show()

image_seg_filename = 'sub-test_fat_label_1_seg.nii.gz'
image_seg = Image(image_seg_filename).change_orientation("RPI")
image_seg = resample_nib(image_seg, new_size=[0.5,0.5,0.5], new_size_type='mm', interpolation='nn', mode='nearest')
image_seg_filename = add_suffix(image_seg_filename, '_resampled')
image_seg.save(image_seg_filename, mutable=True)
plt.imshow(image_seg.data[:,:,image_seg.data.shape[2]//2])
plt.show()

#Get and plot centerlines
image_seg_centerline_pix, image_seg_arr_ctl_pix, image_seg_arr_ctl_der_pix, image_seg_fit_results_pix = get_centerline(image_seg, space='pix')
image_seg_centerline_pix.save(path=add_suffix(image_seg_filename, "_centerline"), mutable=True)
ax = plt.axes(projection='3d')
ax.plot3D(image_seg_arr_ctl_pix[0], image_seg_arr_ctl_pix[1], image_seg_arr_ctl_pix[2])
plt.show()

template_seg_centerline_pix, template_seg_arr_ctl_pix, template_seg_arr_ctl_der_pix, template_seg_fit_results_pix = get_centerline(template_seg, space='pix')
template_seg_centerline_pix.save(path=add_suffix(template_seg_filename, "_centerline"), mutable=True)
ax = plt.axes(projection='3d')
ax.plot3D(template_seg_arr_ctl_pix[0], template_seg_arr_ctl_pix[1], template_seg_arr_ctl_pix[2])
plt.show()

image_seg_centerline_phys, image_seg_arr_ctl_phys, image_seg_arr_ctl_der_phys, image_seg_fit_results_phys = get_centerline(image_seg, space='phys')
template_seg_centerline_phys, template_seg_arr_ctl_phys, template_seg_arr_ctl_der_phys, template_seg_fit_results_phys = get_centerline(template_seg, space='phys')
ax = plt.axes(projection='3d')
ax.plot3D(image_seg_arr_ctl_phys[0], image_seg_arr_ctl_phys[1], image_seg_arr_ctl_phys[2])
ax.plot3D(template_seg_arr_ctl_phys[0], template_seg_arr_ctl_phys[1], template_seg_arr_ctl_phys[2])
plt.show()

#Creating affine transformation matrices to apply to all images
points_src = [[-image_seg_arr_ctl_phys[0, 0], -image_seg_arr_ctl_phys[1, 0], image_seg_arr_ctl_phys[2, 0]], [-image_seg_arr_ctl_phys[0, -1], -image_seg_arr_ctl_phys[1, -1], image_seg_arr_ctl_phys[2, -1]]]
points_dest = [[-template_seg_arr_ctl_phys[0, 0], -template_seg_arr_ctl_phys[1, 0], template_seg_arr_ctl_phys[2, 0]], [-template_seg_arr_ctl_phys[0, -1], -template_seg_arr_ctl_phys[1, -1], template_seg_arr_ctl_phys[2, -1]]]
(rotation_matrix, translation_array, points_moving_reg, points_moving_barycenter) = getRigidTransformFromLandmarks(points_src, points_dest, constraints="Tx_Ty_Tz_Sz", verbose=1)
# writing rigid transformation file
# N.B. x and y dimensions have a negative sign to ensure compatibility between Python and ITK transfo
filename_affine = image_seg_filename.removesuffix('.nii.gz') + '_affine.txt'
text_file = open(filename_affine, 'w')
text_file.write("#Insight Transform File V1.0\n")
text_file.write("#Transform 0\n")
text_file.write("Transform: AffineTransform_double_3_3\n")
text_file.write("Parameters: %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % (
    rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
    rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
    rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2],
    translation_array[0, 0], translation_array[0, 1], translation_array[0, 2]))
text_file.write("FixedParameters: %.9f %.9f %.9f\n" % (points_moving_barycenter[0],
                                                        points_moving_barycenter[1],
                                                        points_moving_barycenter[2]))
text_file.close()

sct_register_multimodal.main(['-i', image_filename, '-iseg', image_seg_filename, '-d', template_filename, '-dseg', template_seg_filename, '-param', 'step=1,type=seg,algo=translation,metric=MeanSquares:step=2,type=seg,algo=rigid,metric=MeanSquares:step=3,type=seg,algo=affine,metric=MeanSquares:step=4,type=seg,algo=bsplinesyn,metric=MeanSquares:step=5,type=seg,algo=columnwise,metric=MeanSquares', '-initwarp', filename_affine])

