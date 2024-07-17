import nibabel as nib
import os
import numpy as np
from scipy import ndimage
from tools import transfo_pix2phys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import run_proc
from scipy.interpolate import InterpolatedUnivariateSpline
from spinalcordtoolbox.registration.landmarks import getRigidTransformFromLandmarks

labels = [1, 2, 3, 4, 5, 6]  # list of labels to produce
fnames = [f'ID{str(i).zfill(4)}' for i in range(1, 77)]  # list of all subjects

### MODIFY THIS PATH
global_path = '/Users/benjamindeleener/data/lumbar/'

# Definitions of paths
data_path = f'{global_path}sourcedata/'  # original data
work_path = f'{global_path}processed/'  # all processed data will be put here
template_path = {label: f'{global_path}template_{label}/' for label in labels}  # templates are created here

# Verify if directories already exist and create them if not
if not os.path.exists(data_path):
    import sys; print('DATA NOT FOUND'); sys.exit()
if not os.path.exists(work_path):
    print(f'Creating directory: {work_path}')
    os.mkdir(work_path)
for label, tpath in template_path.items():
    if not os.path.exists(tpath):
        print(f'Creating directory: {tpath}')
        os.mkdir(tpath)

#############################################################
# Extract centerlines for each label, for each image, and create a main centerline that is the average of all centerlines
# This script assumes the 3rd dimensions is inferior-superior
main_centerlines = {label: {} for label in labels}
for id in fnames:
    image_labels = nib.load(os.path.join(data_path, f"{id}_GT.nii.gz"))
    data_labels = image_labels.get_fdata().astype(np.int8)
    data_centerlines = {label: np.zeros_like(data_labels) for label in labels}
    data_main_centerline = np.zeros_like(data_labels)

    nb_slices = data_labels.shape[2]
    #labels = np.unique(data_labels)

    centerlines_phys = {label: [] for label in labels}
    for label in labels:
        centers = []
        for z in range(nb_slices):
            slice_data = data_labels[:, :, z]
            if label in slice_data:
                com = ndimage.center_of_mass(slice_data == label)
                data_centerlines[label][round(com[0]), round(com[1]), z] = 1  #int(round(label))
                centers.append([int(round(np.mean(com[0]))), int(round(com[1])), z])
        centerlines_phys[label] = transfo_pix2phys(image_labels, centers)

        # if len(centers_slices) == len(labels[1:]):
        #     centers.append((int(round(np.mean([c[0] for c in centers_slices]))), int(round(np.mean([c[1] for c in centers_slices]))), z))
        #     main_centerline_phys.append(transfo_pix2phys(image_labels, [centers[-1]])[0])
        # data_main_centerline[tuple(zip(*centers))] = 1

        main_centerlines[label][id] = centerlines_phys[label]

    for label in labels:
        new_img = image_labels.__class__(data_centerlines[label][:], image_labels.affine, image_labels.header)
        nib.save(new_img, os.path.join(work_path, f"{id}_centerlines_{label}.nii.gz"))
        data_label = data_labels.copy()
        data_label = (data_label == label).astype(np.int_)
        new_img = image_labels.__class__(data_label, image_labels.affine, image_labels.header)
        nib.save(new_img, os.path.join(work_path, f"{id}_mask_{label}.nii.gz"))

for label in labels:
    print(f"Label : {label}")
    ## Generate a template space
    # The template space should have a 0.5 mm^3 isotropic resolution and have enough space to include all structures
    # 1.a Calculate the centers of mass of each main centerline in physical coordinates
    coms = {id: np.average(main_centerlines[label][id], axis=0) for id in fnames}

    # 1.b. Apply a translation on the centerlines to bring them close to each others
    main_centerlines_shifted, ups_and_downs, translation = {}, {}, {}
    for id in fnames:
        main_centerlines_shifted[id] = np.transpose(main_centerlines[label][id] - coms[id])
        ups_and_downs[id] = {'down': main_centerlines_shifted[id][:, 0], 'up': main_centerlines_shifted[id][:, -1]}
        translation[id] = coms[id]

    # Calculating the average length of the centerline. At the moment, length is defined as only the Z difference between the first and last slices
    main_ups_and_downs = {'down': np.average(np.array([ups_and_downs[id]['down'] for id in fnames]), axis=0),
                          'up': np.average(np.array([ups_and_downs[id]['up'] for id in fnames]), axis=0)}
    total_length = abs(main_ups_and_downs['down'][2]) + abs(main_ups_and_downs['up'][2])

    # 1.c Calculating the length of each centerline and applying a translation on the centerlines
    main_centerlines_aligned, zscaling = {}, {}
    for id in fnames:
        own_length = abs(main_centerlines_shifted[id][2, 0]) + abs(main_centerlines_shifted[id][2, -1])
        zscaling[id] = total_length / own_length
        main_centerlines_aligned[id] = main_centerlines_shifted[id].copy()
        main_centerlines_aligned[id][2, :] = main_centerlines_aligned[id][2, :] * zscaling[id]

    # 1.d Calculating average centerline
    centerline_estimators = {}
    number_of_points = 1000
    zi = np.linspace(main_ups_and_downs['down'][2], main_ups_and_downs['up'][2], number_of_points)
    average_centerline = []
    for id in fnames:
        # print(main_centerlines_aligned)
        iux = InterpolatedUnivariateSpline(main_centerlines_aligned[id][2, :], main_centerlines_aligned[id][0, :])
        iuy = InterpolatedUnivariateSpline(main_centerlines_aligned[id][2, :], main_centerlines_aligned[id][1, :])
        centerline_estimators[id] = {'x': iux, 'y': iuy}
        average_centerline.append([[iux(z), iuy(z), z] for z in zi])
    average_centerline = np.average(np.array(average_centerline), axis=0)

    # 1.E Creating affine transformation matrices to apply to all images
    for id in fnames:
        points_src, points_dest = [[-main_centerlines[label][id][0, 0], -main_centerlines[label][id][0, 1], main_centerlines[label][id][0, 2]], [-main_centerlines[label][id][-1, 0], -main_centerlines[label][id][-1, 1], main_centerlines[label][id][-1, 2]]], \
                                  [[-average_centerline[0, 0], -average_centerline[0, 1], average_centerline[0, 2]], [-average_centerline[-1, 0], -average_centerline[-1, 1], average_centerline[-1, 2]]]
        (rotation_matrix, translation_array, points_moving_reg, points_moving_barycenter) = getRigidTransformFromLandmarks(points_src, points_dest, constraints="Tx_Ty_Tz_Sz", verbose=0)
        # writing rigid transformation file
        # N.B. x and y dimensions have a negative sign to ensure compatibility between Python and ITK transfo
        fname_affine = os.path.join(work_path, f'{id}_affine_{label}.txt')
        text_file = open(fname_affine, 'w')
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


    # Visualization of centerlines if needed
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # for id in fnames:
    #     # ax.plot3D(main_centerlines[label][id][0], main_centerlines[label][id][1], main_centerlines[label][id][2])
    #     ax.plot3D(main_centerlines_shifted[id][0], main_centerlines_shifted[id][1], main_centerlines_shifted[id][2])
    #     # ax.plot3D(main_centerlines_aligned[id][0], main_centerlines_aligned[id][1], main_centerlines_aligned[id][2])
    #     # ax.plot3D(average_centerline[:, 0], average_centerline[:, 1], average_centerline[:, 2])
    #
    # ax.set_xlim3d([-100, 100])
    # ax.set_ylim3d([-100, 100])
    # ax.set_zlim3d([-100, 100])
    # plt.show()


    # Create a new space based on the first image, with the appropriate size
    # creating template space
    spacing = 0.5
    size_of_template_space = [300, 300, int((average_centerline[-1, 2] - average_centerline[0, 2])/spacing) + 15]
    template_space = Image(size_of_template_space)
    template_space.data = np.zeros(size_of_template_space)
    template_space.hdr.set_data_dtype('float32')
    center_of_mass = np.average(average_centerline, axis=0)
    origin = [center_of_mass[0] + size_of_template_space[0] * spacing / 2.0,
              center_of_mass[1] - size_of_template_space[1] * spacing / 2.0,
              center_of_mass[2] - size_of_template_space[2] * spacing / 2.0]
    template_space.hdr.as_analyze_map()['dim'] = [3.0, size_of_template_space[0], size_of_template_space[1], size_of_template_space[2], 1.0, 1.0, 1.0, 1.0]
    template_space.hdr.as_analyze_map()['qoffset_x'] = origin[0]
    template_space.hdr.as_analyze_map()['qoffset_y'] = origin[1]
    template_space.hdr.as_analyze_map()['qoffset_z'] = origin[2]
    template_space.hdr.as_analyze_map()['srow_x'][-1] = origin[0]
    template_space.hdr.as_analyze_map()['srow_y'][-1] = origin[1]
    template_space.hdr.as_analyze_map()['srow_z'][-1] = origin[2]
    template_space.hdr.as_analyze_map()['srow_x'][0] = -spacing
    template_space.hdr.as_analyze_map()['srow_y'][1] = spacing
    template_space.hdr.as_analyze_map()['srow_z'][2] = spacing
    template_space.hdr.set_sform(template_space.hdr.get_sform())
    template_space.hdr.set_qform(template_space.hdr.get_sform())
    template_space.save(f"{template_path[label]}template_space_{label}.nii.gz", dtype='float32')


    # generate template centerline as an image
    image_centerline = template_space.copy()
    for coord in average_centerline:
        coord_pix = image_centerline.transfo_phys2pix([coord])[0]
        if 0 <= coord_pix[0] < image_centerline.data.shape[0] and 0 <= coord_pix[1] < image_centerline.data.shape[1] and 0 <= coord_pix[2] < image_centerline.data.shape[2]:
            image_centerline.data[int(coord_pix[0]), int(coord_pix[1]), int(coord_pix[2])] = 1
    image_centerline.save(f"{template_path[label]}template_centerline_{label}.nii.gz", dtype='float32')

    # Align
    for id in fnames: # f"{id}_mask_{label}.nii.gz"
        run_proc(f'sct_apply_transfo -i "{data_path}{id}_T2.nii.gz" -d "{template_path[label]}template_space_{label}.nii.gz" -o "{work_path}{id}_T2_affine_{label}.nii.gz" -w "{work_path}{id}_affine_{label}.txt" -v 1')
        run_proc(f'sct_apply_transfo -i "{data_path}{id}_GT.nii.gz" -d "{template_path[label]}template_space_{label}.nii.gz" -o "{work_path}{id}_GT_affine_{label}.nii.gz" -w "{work_path}{id}_affine_{label}.txt" -x nn -v 1')
        run_proc(f'sct_apply_transfo -i "{work_path}{id}_mask_{label}.nii.gz" -d "{template_path[label]}template_space_{label}.nii.gz" -o "{work_path}{id}_mask_affine_{label}.nii.gz" -w "{work_path}{id}_affine_{label}.txt" -x nn -v 1')


