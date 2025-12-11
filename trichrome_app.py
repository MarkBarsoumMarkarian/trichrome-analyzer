"""
Trichrome Fibrosis Analyzer - Web Interface
============================================

Interactive Streamlit app for analyzing trichrome-stained kidney biopsies.

Usage:
    streamlit run trichrome_app.py
    
Features:
    - Image upload and analysis
    - Real-time segmentation visualization
    - ML-corrected predictions
    - Explainability features (heatmaps, feature contributions)
    - Performance metrics
    - Clinical interpretation
    - PDF report generation
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Import core analyzer
try:
    from trichrome_core import TrichromeAnalyzer
except ImportError:
    st.error("❌ Cannot import trichrome_core.py - make sure it's in the same directory!")
    st.stop()

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Trichrome Fibrosis Analyzer",
    page_icon="🔬",
    layout="wide"
)

# ==================== LOAD MODEL ====================

@st.cache_resource
def load_analyzer():
    """Load analyzer and model"""
    import sys
    import trichrome_core
    
    # Fix pickle module reference for VahadaneNormalizer
    if hasattr(trichrome_core, 'VahadaneNormalizer'):
        sys.modules['__main__'].VahadaneNormalizer = trichrome_core.VahadaneNormalizer
    
    analyzer = TrichromeAnalyzer()
    try:
        analyzer.load_model('trichrome_model.pkl')
        return analyzer, True
    except FileNotFoundError:
        return analyzer, False
    except AttributeError as e:
        # Model file incompatible - needs retraining
        st.sidebar.error(f"⚠️ Model load error: {str(e)}")
        return analyzer, False
    except Exception as e:
        # Catch any other errors
        st.sidebar.error(f"⚠️ Unexpected error: {str(e)}")
        return analyzer, False

analyzer, model_loaded = load_analyzer()

# ==================== SIDEBAR ====================

st.sidebar.title("🔬 Trichrome Analyzer")

if model_loaded:
    st.sidebar.success("✅ ML Model Loaded")
else:
    st.sidebar.warning("⚠️ ML Model Not Found")
    st.sidebar.info("Using raw segmentation only. Train a model using:\n\n"
                   "`python trichrome_core.py train --data training_folder`")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Analysis Options")

show_explainability = st.sidebar.checkbox(
    "🔍 Show Explainability",
    value=False,
    help="Display feature importance and severity heatmap"
)

show_performance = st.sidebar.checkbox(
    "⚡ Show Performance Metrics",
    value=False,
    help="Display processing time breakdown"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About")

with st.sidebar.expander("How it works"):
    st.markdown("""
    **Method:**
    1. Detects blue/green collagen (fibrosis)
    2. Excludes glomeruli (red circles)
    3. Applies ML correction for accuracy
    
    **Stain Colors:**
    - 🔴 Red/Pink = Normal tissue
    - 🔵 Blue/Green = Fibrosis
    - ⚪ White = Glomeruli
    """)

with st.sidebar.expander("Clinical grading"):
    st.markdown("""
    - **Minimal**: <10% fibrosis
    - **Mild**: 10-25% fibrosis
    - **Moderate**: 25-50% fibrosis
    - **Severe**: >50% fibrosis
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Batch Processing")

if st.sidebar.button("Process Folder"):
    st.sidebar.info("Use CLI: `python trichrome_core.py batch --data folder`")

# ==================== MAIN CONTENT ====================

st.title("🔬 Trichrome Fibrosis Analyzer")
st.markdown("**AI-Powered Quantification of Renal Interstitial Fibrosis**")

# File uploader
uploaded_file = st.file_uploader(
    "📁 Upload Trichrome-Stained Image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    help="Upload a Masson's Trichrome-stained kidney biopsy image"
)

if uploaded_file is not None:
    # Load and convert image
    image_pil = Image.open(uploaded_file)
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Display original
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image_pil, use_container_width=True)
        st.caption(f"Size: {image_bgr.shape[1]}×{image_bgr.shape[0]} pixels")
    
    # Analyze
    with st.spinner("🔍 Analyzing image..."):
        if show_performance:
            raw_percent, ml_percent, mask, timings = analyzer.predict(
                image_bgr, return_timings=True
            )
        else:
            raw_percent, ml_percent, mask = analyzer.predict(image_bgr)
            timings = None
    
    # Display segmentation
    with col2:
        st.subheader("🎯 Fibrosis Segmentation")
        st.image(mask, use_container_width=True, clamp=True)
        st.caption("White regions = detected fibrosis")
    
    # Results section
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.metric(
            label="🔢 Raw Segmentation",
            value=f"{raw_percent:.2f}%",
            help="Direct pixel count of blue/green tissue"
        )
    
    with result_col2:
        if ml_percent is not None:
            delta = ml_percent - raw_percent
            st.metric(
                label="🤖 ML-Corrected Prediction",
                value=f"{ml_percent:.2f}%",
                delta=f"{delta:+.2f}%",
                help="Machine learning corrected estimate"
            )
        else:
            st.metric(
                label="🤖 ML-Corrected Prediction",
                value="N/A",
                help="Train a model to enable ML predictions"
            )
    
    with result_col3:
        display_percent = ml_percent if ml_percent is not None else raw_percent
        
        if display_percent < 10:
            grade = "Minimal"
            color = "🟢"
        elif display_percent < 25:
            grade = "Mild"
            color = "🟡"
        elif display_percent < 50:
            grade = "Moderate"
            color = "🟠"
        else:
            grade = "Severe"
            color = "🔴"
        
        st.metric(
            label="🏥 Clinical Grade",
            value=f"{color} {grade}",
            help="Based on fibrosis percentage"
        )
    
    # Clinical interpretation
    st.markdown("---")
    st.subheader("🏥 Clinical Interpretation")
    
    if display_percent < 10:
        st.success(f"**Minimal Fibrosis** ({display_percent:.1f}%): "
                  "Minimal interstitial scarring")
    elif display_percent < 25:
        st.info(f"**Mild Fibrosis** ({display_percent:.1f}%): "
               "Mild interstitial scarring")
    elif display_percent < 50:
        st.warning(f"**Moderate Fibrosis** ({display_percent:.1f}%): "
                  "Moderate interstitial damage")
    else:
        st.error(f"**Severe Fibrosis** ({display_percent:.1f}%): "
                "Extensive interstitial scarring")
    
    # Explainability section
    if show_explainability and model_loaded:
        st.markdown("---")
        st.subheader("🔍 Explainability Analysis")
        
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            st.markdown("**📊 Fibrosis Severity Heatmap**")
            heatmap = analyzer.generate_heatmap(image_bgr, mask)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            st.image(heatmap_rgb, use_container_width=True)
            st.caption("Blue = mild severity | Red = high severity")
        
        with exp_col2:
            st.markdown("**🎯 Top Feature Contributions**")
            
            try:
                explanation = analyzer.explain_features(image_bgr)
                
                if explanation and 'top_features' in explanation:
                    for i, feat in enumerate(explanation['top_features'][:5], 1):
                        contrib = feat.get('contribution', 0)
                        importance = feat.get('importance', 
                                            feat.get('coefficient', 0))
                        
                        with st.container():
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.write(f"**{i}. {feat['name']}**")
                                st.caption(f"Value: {feat['value']:.2f}")
                            with col_b:
                                st.metric("", f"{contrib:.2f}", 
                                         delta=f"{importance:.3f}",
                                         delta_color="off")
                else:
                    st.info("Feature importance not available for this model")
            
            except Exception as e:
                st.error(f"Error generating explanation: {str(e)}")
    
    # Performance metrics
    if show_performance and timings:
        st.markdown("---")
        st.subheader("⚡ Performance Metrics")
        
        perf_cols = st.columns(4)
        
        metrics_list = [
            ('Resize', timings.get('resize', 0)),
            ('Segmentation', timings.get('segmentation', 0)),
            ('ML Inference', timings.get('ml_inference', 0)),
            ('Total', timings.get('total', 0))
        ]
        
        for col, (name, value) in zip(perf_cols, metrics_list):
            with col:
                st.metric(name, f"{value:.1f}ms")
        
        st.caption("Processing time breakdown in milliseconds")
    
    # Download section
    st.markdown("---")
    st.subheader("📥 Download Report")
    
    download_col1, download_col2 = st.columns(2)
    
    with download_col1:
        # Text report
        report_text = f"""TRICHROME FIBROSIS ANALYSIS REPORT
{'='*50}

FILE INFORMATION
Filename: {uploaded_file.name}
Analysis Date: {np.datetime64('today')}
Image Size: {image_bgr.shape[1]}×{image_bgr.shape[0]} pixels

QUANTIFICATION RESULTS
Raw Segmentation: {raw_percent:.2f}%
ML-Corrected Prediction: {ml_percent:.2f}% (if available)

CLINICAL INTERPRETATION
Fibrosis Grade: {grade}
Severity: {display_percent:.2f}%

INTERPRETATION GUIDELINES
- Minimal (<10%): Minimal interstitial scarring
- Mild (10-25%): Mild interstitial scarring
- Moderate (25-50%): Moderate interstitial damage
- Severe (>50%): Extensive interstitial scarring

DISCLAIMER
This analysis is provided for research purposes only and 
should be verified by a qualified pathologist before clinical use.

{'='*50}
Generated by Trichrome Fibrosis Analyzer v2.0
"""
        
        st.download_button(
            label="📄 Download Text Report",
            data=report_text,
            file_name=f"fibrosis_report_{uploaded_file.name}.txt",
            mime="text/plain"
        )
    
    with download_col2:
        # Save segmentation mask
        mask_pil = Image.fromarray(mask)
        import io
        mask_bytes = io.BytesIO()
        mask_pil.save(mask_bytes, format='PNG')
        
        st.download_button(
            label="🖼️ Download Segmentation Mask",
            data=mask_bytes.getvalue(),
            file_name=f"mask_{uploaded_file.name}.png",
            mime="image/png"
        )
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    ⚠️ **CLINICAL DISCLAIMER**: This tool is for research purposes only. 
    Results must be verified by a qualified pathologist and should not be 
    used as the sole basis for clinical decisions.
    """)

else:
    # Landing page
    st.info("👆 Upload a trichrome-stained kidney biopsy image to begin analysis")
    
    st.markdown("---")
    st.subheader("📚 Expected Input Format")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        **Masson's Trichrome Staining:**
        
        This tool analyzes kidney biopsies stained with Masson's Trichrome, 
        where different tissue components appear in distinct colors:
        
        - 🔴 **Red/Pink**: Normal muscle tissue and cytoplasm
        - 🔵 **Blue/Green**: Collagen fibrosis (what we quantify)
        - ⚪ **White/Light**: Glomeruli (excluded from measurement)
        
        The algorithm automatically:
        1. Detects blue/green fibrotic tissue
        2. Identifies and excludes glomeruli
        3. Calculates fibrosis percentage
        4. Applies ML correction (if model available)
        """)
    
    with col_b:
        st.markdown("""
        **Sample Images:**
        
        You can test the tool with:
        - Kidney biopsy images with Masson's Trichrome staining
        - Images should show clear tissue structure
        - Recommended size: 1000×1000 pixels or larger
        - Formats: JPG, PNG, TIF, TIFF
        
        **Training Your Own Model:**
        
        To improve accuracy, train on your own labeled data:
        
        ```bash
        # Name files like: tri_15.5%.jpg, tri_32.1%.jpg
        python trichrome_core.py train --data training_folder
        ```
        
        The ML model learns from your pathologist's annotations.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Trichrome Fibrosis Analyzer v2.0</strong></p>
    <p>For research use only | Validate results with qualified pathologist</p>
</div>
""", unsafe_allow_html=True)