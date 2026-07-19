# Brain Tumour Urgency Score System 

A professional, web-based clinical decision support tool built with Streamlit and Python. This application processes 3D MRI segmentation images (.nii or .nii.gz) to automatically evaluate and quantify brain tumour severity by calculating a combined structural urgency score based on geometric dimensions and shape irregularities.

This feature was developed and integrated into an AI-driven brain tumour segmentation system, serving as the definitive milestone for a university capstone project.

---

## Links

- [Live Demo (Streamlit Community Cloud)](https://brain-tumour-urgency-score-system.streamlit.app/)

---

## Motivation & Purpose

Brain tumour management hinges critically on the speed and accuracy of structural assessment, where slight variations in volume and boundary irregularity heavily influence surgical urgency. Traditionally, extracting precise quantitative metrics from complex 3D radiological data requires extensive manual inspection, which introduces clinical subjectivity and potential triage bottlenecks.

This system was developed to bridge the gap between complex neuroimaging data and real-time clinical decision support. By automating the extraction of 3D geometric dimensions and topological irregularities, the application aims to:
* **Standardise Urgency Evaluation**: Provide objective, repeatable threat stratification scores derived entirely from mathematical morphology rather than qualitative estimation.
* **Accelerate Triage Workflow**: Enable emergency departments and oncology clinics to rapidly interpret segmentation models, ensuring that high-risk cases are immediately flagged for neurosurgical review.
* **Enhance Predictive Visibility**: Offer clinicians clear granular insights into surface invaginations and structural anomalies, which are frequently linked to more aggressive or invasive lesion characteristics.

---

## Tech Stack & Architecture

The application is built on a streamlined, modular architecture designed to handle high-throughput 3D volume data efficiently within a clean, cloud-native frontend environment.

* **Frontend & Presentation**: `Streamlit` — Chosen for its reactive runtime state framework, allowing immediate calculation adjustments and fluid rendering of rich HTML/CSS clinical dashboards.
* **Medical Image Processing**: `Nibabel` — Facilitates seamless low-level parsing of NIfTI-formatted neuroimaging data (`.nii` and `.nii.gz`) and handles coordinate transformation matrices.
* **Scientific Computation & Geometry**: `SciPy` & `NumPy` — Vectorised multi-dimensional array processing allows for ultra-fast spatial spacing calibration and provides the analytical baseline for calculating convex hull volumes.
* **Topological Feature Extraction**: `Scikit-Image` (`skimage`) — Executes advanced 3D Marching Cubes algorithms to extract exact, continuous boundary surface coordinates from complex binary voxel grids.

---

## Challenges & Key Learnings

Building this application exposed several intricate software engineering and clinical data hurdles, leading to critical learning outcomes:

* **Handling Data Scale and Isotropic Inconsistency**: MRI datasets frequently arrive with heterogeneous voxel dimensions, which distorts volume and surface area statistics. To guarantee complete clinical reliability, a data-normalisation layer was engineered to establish strict spatial consistency. 
* **Balancing Technical Architecture with Clinical UX**: Translating raw topological numbers into an intuitive visual format required careful consideration. Developing this tool enhanced key expertise in designing accessible, high-contrast, data-driven layouts that remain highly functional under rapid, high-pressure diagnostic environments.

---

## Streamlit Interface & UI Design

The application features a modern, responsive, and chic medical dashboard interface tailored for clinical readability. 

* **Dual-Zone Layout**: Features a high-contrast theme with a deep slate-grey sidebar (`#1E293B`) for parameter configurations and an elegant off-white main viewport (`#F8FAFC`) designed to minimise eye strain during analysis.
* **Interactive Parameters**: Includes dynamic sidebar controls allowing clinicians to fine-tune probability thresholds and metric weight distributions in real-time.
* **Streamlined File Uploads**: Equipped with a prominent, containerised drag-and-drop zone optimized for large NIfTI datasets.
* **Visual Data Cards**: Metric outputs are dynamically bundled into clean, individual white cards with subtle drop shadows to ensure clear data hierarchy.
* **Color-Coded Urgency Badges**: The final assessment is presented via a prominent, high-visibility notification block that dynamically shifts colours based on the threat level (Red for High Urgency, Orange for Medium, Green for Low).

---

## Urgency Score System & Logic

The core analysis pipeline translates complex 3D morphological features into a singular, actionable clinical metrics framework.

### Data Pre-processing & Binarisation
Upon uploading, the 3D volume data is fetched using `nibabel` and resampled to a 1mm isotropic voxel spacing to ensure strict spatial consistency. The system applies a user-defined **Probability Threshold** to binarise the probability maps, isolating valid tumour voxels from background noise.

### Geometric Shape Metrics (`compute_shape_metrics`)
The system runs advanced topological algorithms via `scikit-image` and `scipy` to extract specific structural benchmarks:
* **Whole Tumour Volume ($cm^3$)**: Calculated directly from total voxel counts and voxel spacing.
* **Surface Area ($mm^2$)**: Computed using a 3D Marching Cubes isosurface extraction algorithm.
* **Sphericity**: Measures how closely the tumour resembles a perfect sphere (where $1.0$ represents a perfect sphere). Lower sphericity indicates a more irregular, invasive growth pattern.
* **Convex Hull Volume ($cm^3$)**: Calculates the volume of the smallest convex shape enclosing the tumour.
* **Convexity**: Derived by dividing the Convex Hull Volume by the actual Tumour Volume. A higher ratio signifies complex surface invaginations and deep structural irregularities.

### Threat Penalisation & Final Score Calculation
To convert these raw geometric attributes into a risk score, the pipeline applies mathematical penalties for morphological anomalies:
* **Sphericity Penalty**: Linearly penalises deviations from a spherical shape:
  $$\text{Sphericity Penalty} = (1.0 - \text{Sphericity}) \times 100$$
* **Convexity Penalty**: Tracks surface complexity. If the tumour is highly irregular, the convex hull significantly exceeds the actual volume, triggering a penalty:
  $$\text{Convexity Penalty} = \max(0.0, \text{Convexity} - 1.0) \times 100$$
* **Composite Shape Score**: Synthesised using a clinical weighting of 70% sphericity influence and 30% convexity irregularity:
  $$\text{Shape Score} = 0.7 \times \text{Sphericity Penalty} + 0.3 \times \text{Convexity Penalty}$$

The **Final Urgency Score** is computed by integrating the weighted volume and shape scores:
$$\text{Final Score} = (\text{Volume Weight} \times \text{Volume}) + (\text{Shape Weight} \times \text{Shape Score})$$

### Clinical Urgency Stratification
The final numerical output maps directly to discrete triage levels:
* **High Urgency** ($\ge 50$): Highlighted in vivid red (`#ff4b4b`), requiring immediate surgical or oncological intervention.
* **Medium Urgency** ($20 - 49$): Highlighted in amber/orange (`#ffa500`), indicating advanced growth requiring expedited scheduling.
* **Low Urgency** ($< 20$): Highlighted in clinical green (`#28a745`), representing stable or early-stage anomalies suitable for standard tracking.

---

## Local Development Setup

Follow these steps to configure your local environment and run the application on your machine.

### Prerequisites
Ensure you have **Python 3.9** or higher installed on your system.

### 1. Clone the Repository
Clone the project repository from GitHub and navigate into the project root directory:
```bash
git clone [https://github.com/ShoheiK12/Brain_Tumour_Urgency_Score_System.git](https://github.com/ShoheiK12/Brain_Tumour_Urgency_Score_System.git)
cd Brain_Tumour_Urgency_Alert_System
```

### 2. Create a Virtual Environment (on Windows)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install streamlit nibabel numpy scipy scikit-image
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## Author

Shohei Kotera