import os
import SimpleITK as sitk

# --- 1. Load Images ---
# moving_seg: Patient segmentation image (label image)
moving_seg = sitk.ReadImage("/g/data/ii16/Image/BrainTumorSeg/Data/BRATS2021/BraTS2021_00003/BraTS2021_00003_seg.nii.gz", sitk.sitkUInt8)
# moving_img: Patient intensity image (used for registration)
moving_img = sitk.ReadImage("/g/data/ii16/Image/BrainTumorSeg/Data/BRATS2021/BraTS2021_00003/BraTS2021_00003_t1ce.nii.gz", sitk.sitkFloat32)


# Get absolute path of the template relative to this script
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "mni_icbm152_t1_tal_nlin_asym_09c.nii")

# fixed: MNI template (intensity image)
fixed = sitk.ReadImage(TEMPLATE_PATH, sitk.sitkFloat32)


# --- 2. Initial Alignment ---
# Align the center of gravity of the moving image to the fixed image
initial_transform = sitk.CenteredTransformInitializer(
    fixed, moving_img, sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)

# --- 3. Registration Setup ---
registration = sitk.ImageRegistrationMethod()
# Use Mattes Mutual Information metric for similarity
registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
# Randomly sample 1% of voxels to speed up metric computation
registration.SetMetricSamplingStrategy(registration.RANDOM)
registration.SetMetricSamplingPercentage(0.01)
# Linear interpolation for intensity images during optimization
registration.SetInterpolator(sitk.sitkLinear)
# Gradient Descent optimizer settings
registration.SetOptimizerAsGradientDescent(learningRate=1.0,
                                           numberOfIterations=100,
                                           convergenceMinimumValue=1e-6,
                                           convergenceWindowSize=10)
registration.SetOptimizerScalesFromPhysicalShift()
# Set initial transform
registration.SetInitialTransform(initial_transform, inPlace=False)

# --- 4. Execute Registration ---
final_transform = registration.Execute(fixed, moving_img)

# --- 5. Resample Segmentation Image to MNI Space ---
# Use nearest neighbor interpolation to preserve label integers
resampled_seg = sitk.Resample(
    moving_seg, fixed, final_transform,
    sitk.sitkNearestNeighbor, 0, moving_seg.GetPixelID()
)

# --- 6. Save Results ---
sitk.WriteImage(resampled_seg, "BraTS2021_00003_seg_in_MNI.nii.gz")
sitk.WriteTransform(final_transform, "BraTS2021_00003_to_MNI.tfm")

print("Registration completed successfully!")