# urgency_score.py

import nibabel as nib
import numpy as np
from scipy.spatial import ConvexHull
from skimage import measure

# 0. Configuration and Data Loading
# --- Voxel Spacing ---
mask_img = nib.load("/g/data/ii16/Image/BrainTumorSeg/Inf_Result/og_test_result/PIFNet_test/BraTS2021_00227_raw_seg.nii.gz")
mask_data = mask_img.get_fdata()
# Force 1mm isotropic spacing for consistent volume/shape calculation
voxel_spacing = (1.0, 1.0, 1.0)
voxel_volume_mm3 = np.prod(voxel_spacing)

print(f"DEBUG: Voxel Spacing: {voxel_spacing}")


# Threshold for binarizing the probability map. 
# Voxels with a tumour probability of 0.5 or higher are considered tumour, which is a common clinical cutoff.
threshold = 0.5
# Binary_mask_data 
mask_data = (mask_data >= threshold).astype(np.uint8)



# Create a binary mask for the whole tumour
whole_mask = (mask_data > 0) 


# Score Weights and Thresholds 
volume_weight = 1.0
shape_weight = 0.3


# 1. Whole Tumour Volume
# Count the number of tumour(not 0 = include tumours)
num_voxels_whole_tumour = np.count_nonzero(whole_mask)
# Calculate the volume of tumour
whole_tumour_vol = (num_voxels_whole_tumour * voxel_volume_mm3) / 1000.0

print(f"Whole tumour volume: {whole_tumour_vol:.2f} cm³")

# 2. Tumor shape metrics (sphericity, convexity)
def compute_shape_metrics(binary_mask, voxel_spacing):
    binary_mask_int = binary_mask.astype(np.int8)
    # Extract the coordinates (x, y, z) of all voxels within the binary_mask and list them up
    coords_vox = np.argwhere(binary_mask_int)
    # Check how many voxels are part of tumour
    num_vox = len(coords_vox)
    
    # Use a dictionary for initial metrics
    if num_vox == 0:
        return {
            "volume_cm3": 0.0, "surface_area_mm2": 0.0, "sphericity": 1.0, 
            "convex_hull_volume_cm3": 0.0, "convexity": 1.0, "shape_score": 0.0,
        }

    # Volume
    volume_mm3 = num_vox * np.prod(voxel_spacing)
    volume_cm3 = volume_mm3 / 1000.0
    
    # Surface Area
    # Initialize surface area
    surface_area_mm2 = 0.0
    try:
        # measure.marching_cubes: extract surface area from 3D voxel data (skinage library)
        verts, faces, _, _ = measure.marching_cubes(
            # np.float32: The input mask is converted to a floating-point type
            # level=0.5: extract boundary between tumour and back(1 and background(0))
            # spacing=voxel_spacing: display coordinates in mm3 
            binary_mask_int, 
            level=0.5, 
            spacing=voxel_spacing
        )
        # Use NumPy's vectorized operations to calculate the area of every triangle and sum them up.
        tris = verts[faces]
        # Calculate the vectors for two sides of each triangle (A and B) and find their cross product.
        cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        # np.linalg.norm(cross, axis=1): Calculates the magnitude (length) of the cross-product vector for every triangle. This magnitude is the area of the parallelogram
        # * 0.5: Area of triangle = half the area of the parallelogram
        surface_area_mm2 = (0.5 * np.linalg.norm(cross, axis=1)).sum()
    except Exception:
        surface_area_mm2 = 0.000001
        

    # Convex Hull Volume
    convex_hull_volume_mm3 = volume_mm3
    # 3D convex hull requires at least four non-coplanar points to form a tetrahedron.
    if num_vox >= 4:
        try:
            # Convert voxel indices into millimeter coordinates to suit actual settings
            coordinates_mm3 = coords_vox * np.array(voxel_spacing)
            # Create the smallest 3D convex shape approximating the tumour's external contour to calculate its volume
            hull = ConvexHull(coordinates_mm3)
            # .volume: an attribute of the ConvexHull object, returning the volume of the solid enclosed by the convex hull -> Computes the enclosed 3D volume of that convex shape
            convex_hull_volume_mm3 = hull.volume
        except Exception:
            pass 

    convex_hull_volume_cm3 = convex_hull_volume_mm3 / 1000.0
    # Assess the irregularity of tumour shape (convexity)
    # max(volume_cm3, 0.000001): Prevent division by zero in case volume_cm3 is zero
    convexity = convex_hull_volume_cm3 / max(volume_cm3, 0.000001)

    # Sphericity
    # Initialize sphericity -> 1.0 = perfect sphere 
    sphericity = 1.0
    # Calculate sphericity unless surface area and volume are negative 
    if surface_area_mm2 > 0 and volume_mm3 > 0:
        # Sphericity = Surface area of an ideal sphere with the same volume as the sphere / Surface area of the actual tumour
        # Formula for calculating the surface area of a sphere of the same volume
        sphericity = (np.pi ** (1.0/3.0)) * ((6.0 * volume_mm3) ** (2.0/3.0)) / surface_area_mm2
        # Prevent the calculation result falling below zero or exceeding one.
        sphericity = np.clip(sphericity, 0.0, 1.0)

    # Shape risk score 
    # 1.0(perfect sphere) - sphericity: the penalty increases the further if the shape deviates from a sphere. Lower sphericity -> higher risk
    sphericity_penalty = (1.0 - sphericity) * 100.0
    # If convexity >= 1, the tumour is smaller than the convex hull and irregular in shape -> high risk. If convexity < 1, no penalty.
    convexity_penalty = max(0.0, convexity - 1.0) * 100.0
    # Synthetic composition with 70% sphericity influence and 30% convexity irregularity.
    shape_score = 0.7 * sphericity_penalty + 0.3 * convexity_penalty
    shape_score = np.clip(shape_score, 0.0, 100.0)

    return {
        "volume_cm3": volume_cm3,
        "surface_area_mm2": surface_area_mm2,
        "sphericity": sphericity,
        "convex_hull_volume_cm3": convex_hull_volume_cm3,
        "convexity": convexity,
        "shape_score": shape_score
    }

shape_metrics = compute_shape_metrics(whole_mask, voxel_spacing)
print("Shape metrics:")
for k, v in shape_metrics.items():
    print(f"  {k}: {v:.2f}")




# 4. Combine volume, and shape into final urgency score
def urgency_score(vol_cm3, shape_score, v_w, s_w):
    # Volume score = Volume × Volume weight -> Larger tumour = higher scores
    volume_score = v_w * vol_cm3
    final_score = volume_score + s_w * shape_score

    if final_score >= 50:
        urgency_level = "High urgency"
    elif final_score >= 20:
        urgency_level = "Medium urgency"
    else:
        urgency_level = "Low urgency"
    
    print(f"Volume score: {volume_score:.2f}, Shape score: {shape_score:.2f}")
    print(f"The urgency score: {final_score:.2f}")
    print(f"Urgency level: {urgency_level}")
    return urgency_level

urgency_level = urgency_score(
    whole_tumour_vol, 
    shape_metrics["shape_score"], 
    volume_weight, shape_weight
)