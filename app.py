# .\.venv\Scripts\python.exe

import streamlit as st
import nibabel as nib
import numpy as np
import tempfile
import os
from scipy.spatial import ConvexHull
from skimage import measure

# 1. ページ基本設定
st.set_page_config(
    page_title="Brain Tumor Urgency Score System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Urgency Score System")
st.write("Upload your 3D MRI segmentation image (.nii.gz) to calculate an urgency score based on tumour volume and shape.")

# 2. サイドバーの設定（パラメータ調整）
st.sidebar.header("⚙️ 解析・スコアパラメータ")

# 閾値
threshold = st.sidebar.slider(
    "腫瘍判定の確率閾値 (Probability Threshold)",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="確率マップを二値化する臨床しきい値。通常は0.5です。"
)

# 重み付け
volume_weight = st.sidebar.slider(
    "体積の重み (Volume Weight)",
    min_value=0.0, max_value=3.0, value=1.0, step=0.1
)

shape_weight = st.sidebar.slider(
    "形状の重み (Shape Weight)",
    min_value=0.0, max_value=3.0, value=0.3, step=0.1
)


# 3. 形状メトリクス計算関数 (元のコードを移植)
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
    
    # 表面積
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
        
    # 凸包体積
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

    # 球形度 (Sphericity)
    sphericity = 1.0
    if surface_area_mm2 > 0 and volume_mm3 > 0:
        sphericity = (np.pi ** (1.0/3.0)) * ((6.0 * volume_mm3) ** (2.0/3.0)) / surface_area_mm2
        sphericity = np.clip(sphericity, 0.0, 1.0)

    # 形状ペナルティ・スコア算出
    sphericity_penalty = (1.0 - sphericity) * 100.0
    convexity_penalty = max(0.0, convexity - 1.0) * 100.0
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

# 4. ファイルアップローダー & メイン処理
uploaded_file = st.file_uploader(
    "Please upload your segmentation result images in NIfTI format (.nii or .nii.gz).", 
    type=["nii", "gz"]
)

if uploaded_file is not None:
    # アップロードファイルを一時的に保存して nibabel で読み込む
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        with st.spinner("🧠 3D画像データを解析中... 数秒かかる場合があります。"):
            # データ読み込みと前処理
            mask_img = nib.load(tmp_file_path)
            mask_data = mask_img.get_fdata()
            
            # 1mm等方性ボクセルで固定
            voxel_spacing = (1.0, 1.0, 1.0)
            voxel_volume_mm3 = np.prod(voxel_spacing)
            
            # 確率しきい値処理
            mask_data_bin = (mask_data >= threshold).astype(np.uint8)
            whole_mask = (mask_data_bin > 0)
            
            # 全腫瘍体積 (cm3)
            num_voxels_whole_tumour = np.count_nonzero(whole_mask)
            whole_tumour_vol = (num_voxels_whole_tumour * voxel_volume_mm3) / 1000.0
            
            # 形状計算
            shape_metrics = compute_shape_metrics(whole_mask, voxel_spacing)
            
        # 一時ファイルの削除
        os.unlink(tmp_file_path)
        
        # 5. スコアの算出と結果の表示
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
        
        # --- 結果表示用レイアウト ---
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 解析結果 (Metrics)")
            st.metric(label="腫瘍体積 (Whole Tumor Volume)", value=f"{whole_tumour_vol:.2f} cm³")
            st.metric(label="球形度 (Sphericity)", value=f"{shape_metrics['sphericity']:.3f} (1.000が完全な球)")
            st.metric(label="表面積 (Surface Area)", value=f"{shape_metrics['surface_area_mm2']:.1f} mm²")
            st.metric(label="凸包体積 (Convex Hull Vol)", value=f"{shape_metrics['convex_hull_volume_cm3']:.2f} cm³")
            st.metric(label="凹凸度 (Convexity)", value=f"{shape_metrics['convexity']:.2f}")

        with col2:
            st.subheader("🚨 Urgency Assessment")
            
            # 緊急度レベルを強調表示
            st.markdown(
                f"""
                <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                    <h2 style="margin: 0; color: white;">{urgency_level}</h2>
                    <p style="margin: 5px 0 0 0; font-size: 1.2em;">総合スコア: <strong>{final_score:.2f}</strong></p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            st.write("")
            st.write(f"**スコア内訳:**")
            st.write(f"- 体積寄与度 (Volume Score): {volume_score:.2f} (重み: {volume_weight})")
            st.write(f"- 形状ペナルティ (Shape Score): {shape_score:.2f} (重み: {shape_weight})")
            
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

else:
    st.info("Please set the parameters on the left and drag and drop a .nii.gz file into the top section to begin the analysis.")