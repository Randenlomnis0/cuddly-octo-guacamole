import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import SimpleITK as sitk
from scipy.ndimage import rotate, zoom
from segment_anything import sam_model_registry, SamPredictor

def resample(volume, spacing, new_spacing=(1.0, 1.0, 1.0)):
    """
    Resample a 3D volume to a new voxel spacing using linear interpolation.

    Parameters
    ----------
    volume : np.ndarray
        Input 3D array.
    spacing : sequence of float
        Original voxel spacing.
    new_spacing : sequence of float, optional
        Target voxel spacing (default: 1.0, 1.0, 1.0).

    Returns
    -------
    np.ndarray
        Resampled volume.
    """
    resize_factor = np.array(spacing) / np.array(new_spacing)
    new_shape = np.round(volume.shape * resize_factor)
    real_resize_factor = new_shape / volume.shape

    resampled = zoom(volume, real_resize_factor, order=1)
    return resampled

def resize_to_cube(volume, target_size=256):
    """
    Center-crop or pad a 3D volume to a cubic shape.

    Parameters
    ----------
    volume : np.ndarray
        Input 3D array.
    target_size : int, optional
        Desired cube size (default: 256).

    Returns
    -------
    np.ndarray
        Cubic volume of shape (target_size, target_size, target_size).
    """
    result = np.zeros((target_size, target_size, target_size), dtype=volume.dtype)

    min_shape = np.minimum(volume.shape, result.shape)

    start_src = [(s - m)//2 for s, m in zip(volume.shape, min_shape)]
    start_dst = [(t - m)//2 for t, m in zip(result.shape, min_shape)]

    result[
        start_dst[0]:start_dst[0]+min_shape[0],
        start_dst[1]:start_dst[1]+min_shape[1],
        start_dst[2]:start_dst[2]+min_shape[2]
    ] = volume[
        start_src[0]:start_src[0]+min_shape[0],
        start_src[1]:start_src[1]+min_shape[1],
        start_src[2]:start_src[2]+min_shape[2]
    ]

    return result

def normalize(img):
    """
    Normalize image to [0, 255].

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Normalized image.
    """
    img = img.astype(np.float32)
    return (((img - img.min()) / (img.max() - img.min() + 1e-8)) * 255).astype(np.uint8)

def dicom_loading_and_visualization():
    ds = pydicom.dcmread("./FORISI/02324177_s2_e_1_BRAIN_DINAMIC_COLINA_AC_FORISI260916")

    n_frames = int(ds.get((0x0028, 0x0008), None).value)
    n_slices = ds.get((0x0054, 0x0081), None).value
    rows = ds.get((0x0028, 0x0010), None).value
    cols = ds.get((0x0028, 0x0011), None).value
    slice_thickness = ds.SliceThickness
    pixel_spacing = ds.PixelSpacing

    pixel_array = ds.pixel_array

    raw_pet_data = pixel_array.reshape(n_frames // n_slices, n_slices, rows, cols)[:, ::-1, :, :]

    pet_data = np.zeros((n_frames // n_slices, 256, 256, 256))

    for i in range(n_frames // n_slices):
        pet_data[i] = normalize(resize_to_cube(resample(raw_pet_data[i], (slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1])))))

    plt.imshow(pet_data[-1][128], cmap='hot')
    plt.title("Last Frame")
    plt.colorbar()
    plt.show()

    avg_frame = np.mean(pet_data, axis=0)

    plt.imshow(avg_frame[128], cmap='hot')
    plt.title("Average PET")
    plt.colorbar()
    plt.show()

    frames = []

    for i in range(n_frames // n_slices):
        img = pet_data[i]

        axial = img[128, :, :]
        coronal = img[:, 128, :]
        sagittal = img[:, :, 128]

        fig, axs = plt.subplots(
            1, 3,
            figsize=(12, 4),
        )
        
        axs[0].imshow(axial, cmap='hot', aspect='auto')
        axs[0].set_title("Axial")
        
        axs[1].imshow(coronal, cmap='hot', aspect='auto')
        axs[1].set_title("Coronal")
        
        axs[2].imshow(sagittal, cmap='hot', aspect='auto')
        axs[2].set_title("Sagittal")

        for ax in axs:
            ax.axis('off')

        fig.canvas.draw()

        frame = np.asarray(fig.canvas.renderer.buffer_rgba())
        frame = frame[:, :, :3]

        frames.append(frame)

        plt.close(fig)

    imageio.mimsave("pet_dynamic_median.gif", frames, duration=1.0)

def rotate_volume(volume, angle, axes=(1, 2)):
    """
    Rotate a 3D volume around given axes.

    Parameters
    ----------
    volume : np.ndarray
        Input 3D array.
    angle : float
        Rotation angle in degrees.
    axes : tuple of int, optional
        Plane of rotation (default: (1, 2)).

    Returns
    -------
    np.ndarray
        Rotated volume.
    """
    return rotate(volume, angle, axes=axes, reshape=False, order=1)

def mip(volume, axis=0):
    """
    Compute a maximum intensity projection (MIP).

    Parameters
    ----------
    volume : np.ndarray
        Input array.
    axis : int, optional
        Projection axis (default: 0).

    Returns
    -------
    np.ndarray
        MIP image.
    """
    return np.max(volume, axis=axis)

def create_mip_gif(mr, pet, output="coregistration.gif"):
    """
    Create a rotating MIP GIF for MR, PET, and their fusion.

    Parameters
    ----------
    mr : np.ndarray
        MR volume.
    pet : np.ndarray
        PET volume.
    output : str, optional
        Output GIF filename.

    Returns
    -------
    None
    """
    frames = []

    for angle in range(0, 360, 10):
        mr_rot = rotate_volume(mr, angle, axes=(1, 2))   # (1, 2)
        pet_rot = rotate_volume(pet, angle, axes=(1, 2)) # (1, 2)

        mr_mip = mip(mr_rot, axis=1)   # 1
        pet_mip = mip(pet_rot, axis=1) # 1

        mr_n = normalize(mr_mip)
        pet_n = normalize(pet_mip)

        fusion = np.zeros((*mr_n.shape, 3), dtype=np.float32)

        fusion[:, :, 0] = mr_n * 0.5
        fusion[:, :, 1] = mr_n * 0.5
        fusion[:, :, 2] = mr_n * 0.5

        fusion[:, :, 0] += pet_n * 0.5

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(mr_mip, cmap='gray')
        axs[0].set_title("MR MIP")

        axs[1].imshow(pet_mip, cmap='hot')
        axs[1].set_title("PET MIP")

        axs[2].imshow(fusion.astype(np.uint8))
        axs[2].set_title("Fusion")

        for ax in axs:
            ax.axis("off")

        fig.canvas.draw()

        frame = np.asarray(fig.canvas.renderer.buffer_rgba())
        frames.append(frame[:, :, :3])

        plt.close(fig)

    imageio.mimsave(output, frames, duration=0.2)

def coregistration_3d():
    ds_pet = pydicom.dcmread("./FORISI/02324177_s2_e_1_BRAIN_DINAMIC_COLINA_AC_FORISI260916")
    ds_mr = pydicom.dcmread("./FORISI/15252129_s1_AX_3D_T1__C_FSPGR_FORISI260916")

    pet_frames = int(ds_pet.get((0x0028, 0x0008), None).value)
    pet_slices = ds_pet.get((0x0054, 0x0081), None).value
    pet_rows = ds_pet.Rows
    pet_cols = ds_pet.Columns
    pet_slice_thickness = ds_pet.SliceThickness
    pet_pixel_spacing = ds_pet.PixelSpacing

    mr_frames = int(ds_mr.get((0x0028, 0x0008), None).value)
    mr_slices = ds_mr.get((0x0054, 0x0081), None).value
    mr_rows = ds_mr.Rows
    mr_cols = ds_mr.Columns
    mr_slice_thickness = ds_mr.SliceThickness
    mr_pixel_spacing = ds_mr.PixelSpacing

    ds_pet = ds_pet.pixel_array.reshape(pet_frames // pet_slices, pet_slices, pet_rows, pet_cols)[:, ::-1, :, :]
    ds_pet = np.mean(ds_pet, axis=0)
    ds_mr = ds_mr.pixel_array.reshape(mr_frames // mr_slices, mr_slices, mr_rows, mr_cols)[-1, ::-1, :, :]

    ds_pet = resample(ds_pet, (pet_slice_thickness, float(pet_pixel_spacing[0]), float(pet_pixel_spacing[1])))
    ds_mr = resample(ds_mr, (mr_slice_thickness, float(mr_pixel_spacing[0]), float(mr_pixel_spacing[1])))

    ds_pet = resize_to_cube(ds_pet)
    ds_mr = resize_to_cube(ds_mr)

    ds_pet = (normalize(ds_pet) / 255).astype(np.float32)
    ds_mr = (normalize(ds_mr) / 255).astype(np.float32)

    fixed = sitk.GetImageFromArray(ds_mr.astype(np.float32))
    moving = sitk.GetImageFromArray(ds_pet.astype(np.float32))

    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    reg = sitk.ImageRegistrationMethod()

    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    # reg.SetMetricAsMeanSquares()
    # reg.SetMetricAsJointHistogramMutualInformation()
    # reg.SetMetricAsCorrelation()

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=1000,
        relaxationFactor=0.5
    )

    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = reg.Execute(fixed, moving)

    print("Final metric value:", reg.GetMetricValue())

    moving_resampled = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID()
    )

    pet_registered = sitk.GetArrayFromImage(moving_resampled)

    create_mip_gif(ds_mr, pet_registered)

def show_bbox(x1, x2, y1, y2, z1, z2, frame):
    """
    Visualize a 3D bounding box on orthogonal slices.

    Parameters
    ----------
    x1, x2, y1, y2, z1, z2 : int
        Bounding box coordinates.
    frame : np.ndarray
        Input 3D volume.

    Returns
    -------
    None
    """
    x_mid = (x1 + x2) // 2
    y_mid = (y1 + y2) // 2
    z_mid = (z1 + z2) // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    img1 = frame[:, x_mid, :]
    axes[0].imshow(img1, cmap='hot')
    axes[0].set_title("View 1: [:, x_mid, :]")

    rect1 = patches.Rectangle(
        (y1, z1),
        y2 - y1,
        z2 - z1,
        linewidth=2,
        edgecolor='cyan',
        facecolor='none'
    )
    axes[0].add_patch(rect1)

    img2 = frame[z_mid, :, :]
    axes[1].imshow(img2, cmap='hot')
    axes[1].set_title("View 2: [z_mid, :, :]")

    rect2 = patches.Rectangle(
        (y1, x1),
        y2 - y1,
        x2 - x1,
        linewidth=2,
        edgecolor='cyan',
        facecolor='none'
    )
    axes[1].add_patch(rect2)

    img3 = frame[:, :, y_mid]
    axes[2].imshow(img3, cmap='hot')
    axes[2].set_title("View 3: [:, :, y_mid]")

    rect3 = patches.Rectangle(
        (x1, z1),
        x2 - x1,
        z2 - z1,
        linewidth=2,
        edgecolor='cyan',
        facecolor='none'
    )
    axes[2].add_patch(rect3)

    plt.tight_layout()
    plt.show()

def image_segmentation():
    ds_pet = pydicom.dcmread("./FORISI/02324177_s2_e_1_BRAIN_DINAMIC_COLINA_AC_FORISI260916")

    n_frames_pet = int(ds_pet.get((0x0028, 0x0008), None).value)
    n_slices_pet = ds_pet.get((0x0054, 0x0081), None).value
    rows_pet = ds_pet.get((0x0028, 0x0010), None).value
    cols_pet = ds_pet.get((0x0028, 0x0011), None).value
    slice_thickness_pet = ds_pet.SliceThickness
    pixel_spacing_pet = ds_pet.PixelSpacing

    pixel_array_pet = ds_pet.pixel_array

    raw_pet_data = pixel_array_pet.reshape(n_frames_pet // n_slices_pet, n_slices_pet, rows_pet, cols_pet)[:, ::-1, :, :]

    pet_data = np.zeros((n_frames_pet // n_slices_pet, 256, 256, 256))

    for i in range(n_frames_pet // n_slices_pet):
        pet_data[i] = resize_to_cube(resample(raw_pet_data[i], (slice_thickness_pet, float(pixel_spacing_pet[0]), float(pixel_spacing_pet[1]))))

    x1 = 140
    x2 = 187
    y1 = 140
    y2 = 178
    z1 = 110
    z2 = 150

    show_bbox(x1, x2, y1, y2, z1, z2, pet_data[-1])

    ds_mr = pydicom.dcmread("./FORISI/15252129_s1_AX_3D_T1__C_FSPGR_FORISI260916")

    mr_frames = int(ds_mr.get((0x0028, 0x0008), None).value)
    mr_slices = ds_mr.get((0x0054, 0x0081), None).value
    mr_rows = ds_mr.Rows
    mr_cols = ds_mr.Columns
    mr_slice_thickness = ds_mr.SliceThickness
    mr_pixel_spacing = ds_mr.PixelSpacing

    ds_mr = ds_mr.pixel_array.reshape(mr_frames // mr_slices, mr_slices, mr_rows, mr_cols)[-1, ::-1, :, :]
    ds_mr = resample(ds_mr, (mr_slice_thickness, float(mr_pixel_spacing[0]), float(mr_pixel_spacing[1])))
    ds_mr = resize_to_cube(ds_mr)

    for slice in ds_mr:
        slice = normalize(slice)

    x1 = 155
    x2 = 205
    y1 = 148
    y2 = 192
    z1 = 115
    z2 = 159

    show_bbox(x1, x2, y1, y2, z1, z2, ds_mr)

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    model.to("cpu")

    predictor = SamPredictor(model)

    segmentation = np.zeros_like(ds_mr)

    for z in np.linspace(z1, z2-1, 5).astype(int):
        img = normalize(ds_mr[z])
        img_rgb = np.stack([img]*3, axis=-1)

        predictor.set_image(img_rgb)

        box = np.array([x1, y1, x2, y2])

        masks, _, _ = predictor.predict(
            box=box[None, :],
            multimask_output=False
        )

        segmentation[z] = masks[0]

    plt.figure(figsize=(6, 6))

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    for i, z in enumerate(np.linspace(z1, z2-1, 5).astype(int)):
        axes[i].imshow(ds_mr[z], cmap='gray')
        axes[i].imshow(segmentation[z], cmap='jet', alpha=0.4)
        axes[i].set_title(f"z={z}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # dicom_loading_and_visualization()
    # coregistration_3d()
    image_segmentation()

if __name__ == '__main__':
    main()
