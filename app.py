# .\.venv\Scripts\python.exe

import streamlit as st
import nibabel as nib
import numpy as np
import tempfile
import os
from scipy.spatial import ConvexHull
from skimage import measure

# Page settings
st.set_page_config(
    page_title="Brain Tumor Urgency Score System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Urgency Score System")
st.write("Upload your 3D MRI segmentation image (.nii.gz) to calculate an urgency score based on tumour volume and shape.")

# 2. Sidebar settings
st.sidebar.header("⚙️ Analysis parameter")

threshold = st.sidebar.slider(
    "Probability Threshold",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="Clinical threshold for binarising probability maps. Voxels with a tumour probability of 0.5 or higher are considered tumour, which is a common clinical cutoff."
)

volume_weight = st.sidebar.slider(
    "Volume Weight",
    min_value=0.0, max_value=3.0, value=1.0, step=0.1
)

shape_weight = st.sidebar.slider(
    "Shape Weight",
    min_value=0.0, max_value=3.0, value=0.3, step=0.1
)


# Shape metrics
def compute_shape_metrics(binary_mask, voxel_spacing):
    binary_mask_int = binary_mask.astype(np.int8)
    coords_vox = np.argwhere(binary_mask_int)
    num_vox = len(coords_vox)
    
    if num_vox == 0:
        return {
            "volume_cm3": 0.0, "surface_area_mm2": 0.0, "sphericity": 1.0, 
            "convex_hull_volume_cm3": 0.0, "convexity": 1.0, "shape_score": 0.0,
        }

    volume_mm3 = num_vox * np.prod(voxel_spacing)
    volume_cm3 = volume_mm3 / 1000.0
    
    # Surface area
    surface_area_mm2 = 0.0
    try:
        verts, faces, _, _ = measure.marching_cubes(
            binary_mask_int, 
            level=0.5, 
            spacing=voxel_spacing
        )
        tris = verts[faces]
        cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        surface_area_mm2 = (0.5 * np.linalg.norm(cross, axis=1)).sum()
    except Exception:
        surface_area_mm2 = 0.000001
        
    # Convex hull volume
    convex_hull_volume_mm3 = volume_mm3
    if num_vox >= 4:
        try:
            coordinates_mm3 = coords_vox * np.array(voxel_spacing)
            hull = ConvexHull(coordinates_mm3)
            convex_hull_volume_mm3 = hull.volume
        except Exception:
            pass 

    convex_hull_volume_cm3 = convex_hull_volume_mm3 / 1000.0
    convexity = convex_hull_volume_cm3 / max(volume_cm3, 0.000001)

    # Sphericity
    sphericity = 1.0
    if surface_area_mm2 > 0 and volume_mm3 > 0:
        sphericity = (np.pi ** (1.0/3.0)) * ((6.0 * volume_mm3) ** (2.0/3.0)) / surface_area_mm2
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

# File uploader & Main process
uploaded_file = st.file_uploader(
    "Please upload your segmentation result images in NIfTI format (.nii or .nii.gz).", 
    type=["nii", "gz"]
)

if uploaded_file is not None:
    # Save uploaded file temporarily and read it by nibabel
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        with st.spinner("🧠 Analysing 3D image data... This may take a few seconds."):
            # Load data and pre-process
            mask_img = nib.load(tmp_file_path)
            mask_data = mask_img.get_fdata()
            
            # Force 1mm isotropic spacing for consistent volume/shape calculation
            voxel_spacing = (1.0, 1.0, 1.0)
            voxel_volume_mm3 = np.prod(voxel_spacing)
            
            # Binary_mask_data 
            mask_data_bin = (mask_data >= threshold).astype(np.uint8)
            whole_mask = (mask_data_bin > 0)
            
            # Whole tumour volume
            num_voxels_whole_tumour = np.count_nonzero(whole_mask)
            whole_tumour_vol = (num_voxels_whole_tumour * voxel_volume_mm3) / 1000.0
            
            # Shape metrics
            shape_metrics = compute_shape_metrics(whole_mask, voxel_spacing)
            
        # Delete temporary file
        os.unlink(tmp_file_path)
        
        # 5. Calculate score and Display the result
        vol_cm3 = whole_tumour_vol
        shape_score = shape_metrics["shape_score"]
        
        volume_score = volume_weight * vol_cm3
        final_score = volume_score + shape_weight * shape_score

        if final_score >= 50:
            urgency_level = "High urgency"
            bg_color = "#ff4b4b"
        elif final_score >= 20:
            urgency_level = "Medium urgency"
            bg_color = "#ffa500"
        else:
            urgency_level = "Low urgency"
            bg_color = "#28a745"

        st.success("Analysis complete!")
        
        # Layout for result
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 The result of analysis (Metrics)")
            st.metric(label="Whole Tumor Volume", value=f"{whole_tumour_vol:.2f} cm³")
            st.metric(label="Sphericity", value=f"{shape_metrics['sphericity']:.3f} (1.000 is perfect sphere)")
            st.metric(label="Surface Area", value=f"{shape_metrics['surface_area_mm2']:.1f} mm²")
            st.metric(label="Convex Hull Volume", value=f"{shape_metrics['convex_hull_volume_cm3']:.2f} cm³")
            st.metric(label="Convexity", value=f"{shape_metrics['convexity']:.2f}")

        with col2:
            st.subheader("🚨 Urgency Assessment")
            
            # Highlight the urgency level
            st.markdown(
                f"""
                <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                    <h2 style="margin: 0; color: white;">{urgency_level}</h2>
                    <p style="margin: 5px 0 0 0; font-size: 1.2em;">Final score: <strong>{final_score:.2f}</strong></p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            st.write("")
            st.write(f"**Score breakdown:**")
            st.write(f"- Volume Score: {volume_score:.2f} (Parameter: {volume_weight})")
            st.write(f"- Shape Score: {shape_score:.2f} (Parameter: {shape_weight})")
            
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

else:
    st.info("Please set the parameters on the left and drag and drop a .nii.gz file into the top section to begin the analysis.")