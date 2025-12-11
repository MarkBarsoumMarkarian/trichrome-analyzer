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
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, tissue_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        lower_blue_green = np.array([75, 15, 30])
        upper_blue_green = np.array([170, 255, 255])
        all_blue_green = cv2.inRange(hsv, lower_blue_green, upper_blue_green)
        all_blue_green = cv2.bitwise_and(all_blue_green, tissue_mask)
        
        red_glomeruli = self.detect_red_pink_glomeruli(image)
        kernel_expand = np.ones((100, 100), np.uint8)
        red_glomeruli_expanded = cv2.dilate(red_glomeruli.astype(np.uint8) * 255, kernel_expand, iterations=1)
        roi_blue_green = cv2.bitwise_and(all_blue_green, all_blue_green, mask=red_glomeruli_expanded)
        
        fibrosis_mask = cv2.bitwise_or(all_blue_green, roi_blue_green)
        
        blue_glomeruli = self.detect_blue_green_glomeruli(image)
        fibrosis_mask[blue_glomeruli] = 0
        fibrosis_mask[red_glomeruli] = 0
        
        light_mask = gray > 190
        fibrosis_mask[light_mask] = 0
        
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_medium = np.ones((4, 4), np.uint8)
        fibrosis_mask = cv2.morphologyEx(fibrosis_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        fibrosis_mask = cv2.morphologyEx(fibrosis_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        
        return fibrosis_mask
    
    def extract_features(self, image, mask):
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
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
    
    def predict(self, image):
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
        
        return raw_percent, ml_prediction, fibrosis_mask, image

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

uploaded_file = st.file_uploader("Upload Trichrome Image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    raw_percent, ml_percent, mask, processed_image = analyzer.predict(image_bgr)

    st.image(mask)

    display_percent = ml_percent if ml_percent else raw_percent

    # ✅ FIXED PDF BUTTON
    if st.button("📥 Generate PDF Report"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        title = Paragraph("<b>TRICHROME FIBROSIS ANALYSIS REPORT</b>", ParagraphStyle(name="TitleStyle", alignment=TA_CENTER, fontSize=16))
        elements.append(title)
        elements.append(Spacer(1, 12))

        elements.append(Paragraph(f"<b>Image:</b> {uploaded_file.name}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        results_table = Table([
            ["Metric", "Value"],
            ["Raw Segmentation", f"{raw_percent:.2f}%"],
            ["ML Prediction", f"{ml_percent:.2f}%" if ml_percent else "N/A"],
            ["Image Size", f"{processed_image.shape[1]} × {processed_image.shape[0]} px"],
            ["Fibrosis Grade", "Minimal" if display_percent < 10 else "Mild" if display_percent < 25 else "Moderate" if display_percent < 50 else "Severe"]
        ])

        results_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, rl_colors.black),
            ('BACKGROUND', (0,0), (-1,0), rl_colors.lightgrey),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ]))

        elements.append(results_table)
        elements.append(Spacer(1, 20))
        disclaimer = Paragraph("<b>Disclaimer:</b> This analysis is for research purposes only.", styles["Normal"])
        elements.append(disclaimer)

        doc.build(elements)
        buffer.seek(0)

        st.download_button(
            label="✅ Download PDF Report",
            data=buffer,
            file_name=f"fibrosis_report_{uploaded_file.name}.pdf",
            mime="application/pdf"
        )
