import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Page config
st.set_page_config(
    page_title="Trichrome Fibrosis Analyzer",
    page_icon="🔬",
    layout="wide"
)

# --- TrichromeAnalyzer Core Methods (Embedded) ---
class TrichromeAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
    
    def detect_red_pink_glomeruli(self, image):
        """Detect RED/PINK glomeruli (healthy glomeruli)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        lower_red1 = np.array([0, 20, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 20, 50])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_pink_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        blurred = cv2.GaussianBlur(red_pink_mask, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
            param1=50, param2=30, minRadius=15, maxRadius=120
        )
        
        red_glomeruli_mask = np.zeros(gray.shape, dtype=np.uint8)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = int(circle[2] * 1.1)
                cv2.circle(red_glomeruli_mask, center, radius, 255, -1)
        
        return red_glomeruli_mask > 0
    
    def detect_blue_green_glomeruli(self, image):
        """Detect BLUE/GREEN glomeruli (sclerosed/abnormal)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        lower_blue_green = np.array([80, 30, 50])
        upper_blue_green = np.array([170, 255, 255])
        blue_green_mask = cv2.inRange(hsv, lower_blue_green, upper_blue_green)
        
        blurred = cv2.GaussianBlur(blue_green_mask, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
            param1=50, param2=30, minRadius=15, maxRadius=120
        )
        
        blue_glomeruli_mask = np.zeros(gray.shape, dtype=np.uint8)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = int(circle[2] * 1.1)
                cv2.circle(blue_glomeruli_mask, center, radius, 255, -1)
        
        return blue_glomeruli_mask > 0
    
    def segment_fibrosis(self, image):
        """Segment fibrosis using combined approach (A + B)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, tissue_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # OPTION A: Detect ALL blue/green tissue
        lower_blue_green = np.array([75, 15, 30])
        upper_blue_green = np.array([170, 255, 255])
        all_blue_green = cv2.inRange(hsv, lower_blue_green, upper_blue_green)
        all_blue_green = cv2.bitwise_and(all_blue_green, tissue_mask)
        
        # OPTION B: Expanded ROI around red glomeruli
        red_glomeruli = self.detect_red_pink_glomeruli(image)
        kernel_expand = np.ones((100, 100), np.uint8)
        red_glomeruli_expanded = cv2.dilate(red_glomeruli.astype(np.uint8) * 255, 
                                            kernel_expand, iterations=1)
        roi_blue_green = cv2.bitwise_and(all_blue_green, all_blue_green,
                                         mask=red_glomeruli_expanded)
        
        # COMBINE both approaches
        fibrosis_mask = cv2.bitwise_or(all_blue_green, roi_blue_green)
        
        # EXCLUSIONS
        blue_glomeruli = self.detect_blue_green_glomeruli(image)
        fibrosis_mask[blue_glomeruli] = 0
        fibrosis_mask[red_glomeruli] = 0
        
        light_mask = gray > 190
        fibrosis_mask[light_mask] = 0
        
        # Cleanup
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_medium = np.ones((4, 4), np.uint8)
        fibrosis_mask = cv2.morphologyEx(fibrosis_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        fibrosis_mask = cv2.morphologyEx(fibrosis_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        
        return fibrosis_mask
    
    def extract_features(self, image, mask):
        """Extract 14 features for ML model"""
        features = []
        
        total_pixels = mask.size
        fibrosis_pixels = np.sum(mask > 0)
        fibrosis_ratio = fibrosis_pixels / total_pixels
        features.append(fibrosis_ratio * 100)
        
        if fibrosis_pixels > 0:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            h_vals = hsv[:, :, 0][mask > 0]
            s_vals = hsv[:, :, 1][mask > 0]
            v_vals = hsv[:, :, 2][mask > 0]
            
            features.extend([
                np.mean(h_vals), np.std(h_vals),
                np.mean(s_vals), np.std(s_vals),
                np.mean(v_vals),
            ])
            
            b_vals = lab[:, :, 2][mask > 0]
            features.extend([np.mean(b_vals), np.std(b_vals)])
            
            y_coords, x_coords = np.where(mask > 0)
            features.extend([
                np.std(x_coords) / image.shape[1],
                np.std(y_coords) / image.shape[0],
            ])
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            num_regions = num_labels - 1
            
            if num_regions > 0:
                areas = stats[1:, cv2.CC_STAT_AREA]
                features.extend([
                    num_regions,
                    np.mean(areas),
                    np.std(areas) if len(areas) > 1 else 0,
                    np.max(areas),
                ])
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0] * 13)
        
        return np.array(features)
    
    def load_model(self, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
    
    def predict(self, image):
        """Predict fibrosis percentage"""
        # Resize if needed
        max_dim = 1024
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        fibrosis_mask = self.segment_fibrosis(image)
        raw_percent = (np.sum(fibrosis_mask > 0) / fibrosis_mask.size) * 100
        
        if self.model is not None:
            features = self.extract_features(image, fibrosis_mask)
            features_scaled = self.scaler.transform([features])
            ml_prediction = self.model.predict(features_scaled)[0]
            ml_prediction = np.clip(ml_prediction, 0, 100)
        else:
            ml_prediction = None
        
        return raw_percent, ml_prediction, fibrosis_mask
    
    def predict_optimized(self, image):
        """Predict with performance timing"""
        import time
        timings = {}
        total_start = time.time()
        
        # Resize if needed
        t0 = time.time()
        max_dim = 1024
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        timings['load_resize'] = (time.time() - t0) * 1000
        
        # Segmentation
        t0 = time.time()
        fibrosis_mask = self.segment_fibrosis(image)
        timings['segmentation'] = (time.time() - t0) * 1000
        
        # Features
        t0 = time.time()
        raw_percent = (np.sum(fibrosis_mask > 0) / fibrosis_mask.size) * 100
        features = self.extract_features(image, fibrosis_mask)
        timings['features'] = (time.time() - t0) * 1000
        
        # ML inference
        t0 = time.time()
        if self.model is not None:
            features_scaled = self.scaler.transform([features])
            ml_prediction = self.model.predict(features_scaled)[0]
            ml_prediction = np.clip(ml_prediction, 0, 100)
        else:
            ml_prediction = None
        timings['ml_inference'] = (time.time() - t0) * 1000
        
        timings['total'] = (time.time() - total_start) * 1000
        
        return raw_percent, ml_prediction, fibrosis_mask, timings
    
    def generate_contribution_heatmap(self, image, fibrosis_mask):
        """Create severity heatmap"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        heatmap = np.zeros_like(image, dtype=np.float32)
        mask_bool = fibrosis_mask > 0
        
        if np.any(mask_bool):
            sat_normalized = np.zeros_like(saturation, dtype=np.uint8)
            sat_values = saturation[mask_bool]
            if sat_values.max() > sat_values.min():
                sat_normalized[mask_bool] = ((saturation[mask_bool] - sat_values.min()) / 
                                             (sat_values.max() - sat_values.min()) * 255).astype(np.uint8)
        
            heatmap_colored = cv2.applyColorMap(sat_normalized, cv2.COLORMAP_JET)
            heatmap[mask_bool] = heatmap_colored[mask_bool]
        
        overlay = cv2.addWeighted(image.astype(np.float32), 0.6, heatmap, 0.4, 0)
        return overlay.astype(np.uint8)
    
    def explain_features(self, features, prediction):
        """Generate feature importance explanation"""
        feature_names = [
            'Area %', 'Hue Mean', 'Hue Std', 'Sat Mean', 'Sat Std',
            'Value Mean', 'LAB_B Mean', 'LAB_B Std', 'Spread X', 
            'Spread Y', 'Num Regions', 'Avg Region Size', 
            'Region Size Std', 'Max Region Size'
        ]
        
        explanation = {
            'prediction': prediction,
            'features': {}
        }
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            for name, feat_val, importance in zip(feature_names, features, importances):
                explanation['features'][name] = {
                    'value': float(feat_val),
                    'importance': float(importance),
                    'contribution': float(feat_val * importance)
                }
        elif hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_
            for name, feat_val, coef in zip(feature_names, features, coefficients):
                explanation['features'][name] = {
                    'value': float(feat_val),
                    'coefficient': float(coef),
                    'contribution': float(feat_val * coef)
                }
        
        sorted_features = sorted(
            explanation['features'].items(),
            key=lambda x: abs(x[1].get('contribution', 0)),
            reverse=True
        )
        
        explanation['top_features'] = [
            {'name': name, **data} 
            for name, data in sorted_features[:5]
        ]
        
        return explanation


# --- Load Model ---
@st.cache_resource
def load_analyzer():
    analyzer = TrichromeAnalyzer()
    try:
        analyzer.load_model('trichrome_fibrosis_model.pkl')
        return analyzer, True
    except FileNotFoundError:
        return analyzer, False

analyzer, model_loaded = load_analyzer()

# --- UI ---
st.title("🔬 Trichrome Fibrosis Analyzer")
st.markdown("**AI-Powered Quantification of Renal Interstitial Fibrosis**")

if model_loaded:
    st.sidebar.success("✅ ML Model Loaded")
else:
    st.sidebar.warning("⚠️ ML Model Not Found (Using Raw Segmentation Only)")

st.sidebar.markdown("---")
st.sidebar.markdown("### Analysis Options")

show_explainability = st.sidebar.checkbox("🔍 Show Explainability", value=False,
                                          help="Display feature importance and severity heatmap")
show_performance = st.sidebar.checkbox("⚡ Show Performance Metrics", value=False,
                                      help="Display processing time breakdown")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This tool analyzes Masson's Trichrome-stained kidney biopsies to quantify interstitial fibrosis.

**Method:**
- Detects blue/green collagen (fibrosis)
- Excludes glomeruli 
- Uses ML correction for accuracy

**Features:**
- 🔍 Explainable AI
- ⚡ Real-time processing
- 📊 Clinical validation ready
- 📥 PDF reports
""")

# File upload
uploaded_file = st.file_uploader(
    "Upload Trichrome Image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    help="Upload a trichrome-stained kidney biopsy image"
)

if uploaded_file is not None:
    # Load image
    image_pil = Image.open(uploaded_file)
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Create columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image_pil, use_container_width=True)
    
    # Analyze
    with st.spinner("Analyzing image..."):
        if show_performance:
            raw_percent, ml_percent, mask, timings = analyzer.predict_optimized(image_bgr)
        else:
            raw_percent, ml_percent, mask = analyzer.predict(image_bgr)
            timings = None
    
    with col2:
        st.subheader("Fibrosis Segmentation")
        st.image(mask, use_container_width=True, clamp=True)
    
    # Explainability features
    if show_explainability and model_loaded:
        st.markdown("---")
        st.subheader("🔍 Explainability Analysis")
        
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            # Severity heatmap
            st.markdown("**Fibrosis Severity Heatmap**")
            heatmap = analyzer.generate_contribution_heatmap(image_bgr, mask)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            st.image(heatmap_rgb, use_container_width=True,
                    caption="Color intensity shows fibrosis severity (blue=mild, red=severe)")
        
        with exp_col2:
            # Feature explanation
            st.markdown("**Feature Contributions**")
            fibrosis_mask = analyzer.segment_fibrosis(image_bgr)
            features = analyzer.extract_features(image_bgr, fibrosis_mask)
            explanation = analyzer.explain_features(features, ml_percent if ml_percent else raw_percent)
            
            if explanation and 'top_features' in explanation:
                for feat in explanation['top_features'][:5]:
                    contrib = feat.get('contribution', 0)
                    importance = feat.get('importance', feat.get('coefficient', 0))
                    
                    col_name, col_value = st.columns([2, 1])
                    with col_name:
                        st.write(f"**{feat['name']}**")
                    with col_value:
                        st.metric("", f"{contrib:.2f}", 
                                 delta=f"{importance:.3f}",
                                 delta_color="off")
    
    # Performance metrics
    if show_performance and timings:
        st.markdown("---")
        st.subheader("⚡ Performance Metrics")
        
        perf_cols = st.columns(5)
        metrics_list = [
            ('Load/Resize', timings.get('load_resize', 0)),
            ('Segmentation', timings.get('segmentation', 0)),
            ('Features', timings.get('features', 0)),
            ('ML Inference', timings.get('ml_inference', 0)),
            ('Total', timings.get('total', 0))
        ]
        
        for col, (name, value) in zip(perf_cols, metrics_list):
            with col:
                st.metric(name, f"{value:.1f}ms")
    
    # Results
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.metric(
            label="Raw Segmentation",
            value=f"{raw_percent:.2f}%",
            help="Direct pixel count of blue/green tissue"
        )
    
    with result_col2:
        if ml_percent is not None:
            st.metric(
                label="ML-Corrected Prediction",
                value=f"{ml_percent:.2f}%",
                delta=f"{ml_percent - raw_percent:+.2f}%",
                help="Machine learning corrected estimate"
            )
        else:
            st.metric(
                label="ML-Corrected Prediction",
                value="N/A",
                help="Model not loaded"
            )
    
    with result_col3:
        st.metric(
            label="Image Size",
            value=f"{image_bgr.shape[1]}×{image_bgr.shape[0]}",
            help="Width × Height in pixels"
        )
    
    # Clinical interpretation
    st.markdown("---")
    st.subheader("🏥 Clinical Interpretation")
    
    display_percent = ml_percent if ml_percent is not None else raw_percent
    
    if display_percent < 10:
        st.success("**Minimal Fibrosis** (<10%): Minimal interstitial scarring")
    elif display_percent < 25:
        st.info("**Mild Fibrosis** (10-25%): Mild interstitial scarring")
    elif display_percent < 50:
        st.warning("**Moderate Fibrosis** (25-50%): Moderate interstitial damage")
    else:
        st.error("**Severe Fibrosis** (>50%): Extensive interstitial scarring")
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    ⚠️ **Clinical Disclaimer**: This tool is for research purposes only. 
    Results must be verified by a qualified pathologist and should not be used 
    as the sole basis for clinical decisions.
    """)
    
    # Download report
    if st.button("📥 Download Report"):
        report = f"""TRICHROME FIBROSIS ANALYSIS REPORT
=====================================

Image: {uploaded_file.name}
Analysis Date: {np.datetime64('today')}

RESULTS
-------
Raw Segmentation: {raw_percent:.2f}%
ML-Corrected: {ml_percent:.2f}% (if available)
Image Dimensions: {image_bgr.shape[1]}×{image_bgr.shape[0]} pixels

INTERPRETATION
--------------
Fibrosis Grade: {"Minimal" if display_percent < 10 else "Mild" if display_percent < 25 else "Moderate" if display_percent < 50 else "Severe"}

DISCLAIMER
----------
This analysis is provided for research purposes only and should be 
verified by a qualified pathologist before clinical use.

Generated by Trichrome Fibrosis Analyzer v1.1.0
"""
        st.download_button(
            label="Download as TXT",
            data=report,
            file_name=f"fibrosis_report_{uploaded_file.name}.txt",
            mime="text/plain"
        )
else:
    st.info("👆 Upload a trichrome-stained image to begin analysis")
    
    # Show example
    st.markdown("---")
    st.subheader("Expected Input")
    st.markdown("""
    **Trichrome Staining:**
    - 🔴 Red/Pink = Normal muscle tissue (cytoplasm)
    - 🔵 Blue/Green = Collagen fibrosis (what we measure)
    - ⚪ White circles = Glomeruli (excluded from calculation)
    """)
