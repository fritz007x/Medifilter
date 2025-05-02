#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import streamlit as st
import nibabel as nib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
from scipy import ndimage
import matplotlib.cm as cm
import dog as dog_filter
import filters
from pathlib import Path

st.set_page_config(
    page_title="MediFilter: Advanced Medical Imaging Filters",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a more professional look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #4e73df;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #3756a4;
    }
    .filter-card {
        background-color: white;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .results-section {
        background-color: white;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stSlider > div > div {
        background-color: #4e73df;
    }
</style>
""", unsafe_allow_html=True)


def normalize_image_for_display(image_data):
    """Normalize image data for display."""
    img_min = np.min(image_data)
    img_max = np.max(image_data)
    if img_max > img_min:
        return (image_data - img_min) / (img_max - img_min)
    return image_data


def plot_slice(img_data, title, colormap='gray'):
    """Plot a slice of 3D data or 2D image."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if img_data.ndim == 3:
        # For 3D data, show middle slice
        middle_slice_idx = img_data.shape[2] // 2
        slice_data = img_data[:, :, middle_slice_idx]
    else:
        # For 2D data
        slice_data = img_data
    
    # Normalize for display
    normalized_data = normalize_image_for_display(slice_data)
    
    # Display image
    ax.imshow(normalized_data, cmap=colormap)
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    return fig


def apply_filter(image_data, filter_type, params):
    """Apply the selected filter to the image data."""
    if filter_type == "DoG (Difference of Gaussian)":
        # Create temporary NIfTI image for DoG filter
        if isinstance(image_data, nib.Nifti1Image):
            img = image_data
        else:
            # If it's a numpy array, create a NIfTI image with identity affine
            affine = np.eye(4)
            img = nib.Nifti1Image(image_data, affine)
        
        # Apply DoG filter
        fwhm = params.get("fwhm", 3)
        verbose = params.get("verbose", 0)
        result = dog_filter.dog_img(img, fwhm=fwhm, verbose=verbose)
        
        # Convert back to numpy array
        if isinstance(image_data, nib.Nifti1Image):
            return result
        else:
            return result.get_fdata()
    
    elif filter_type == "Frangi (Vessel Enhancement)":
        scale_range = (params.get("scale_min", 1), params.get("scale_max", 10))
        scale_step = params.get("scale_step", 2)
        beta = params.get("beta", 0.5)
        black_ridges = params.get("black_ridges", True)
        use_itk = params.get("use_itk", True)
        verbose = params.get("verbose", 0)
        
        return filters.frangi_filter(
            image_data, 
            scale_range=scale_range, 
            scale_step=scale_step, 
            beta=beta, 
            black_ridges=black_ridges,
            use_itk=use_itk,
            verbose=verbose
        )
    
    elif filter_type == "Sobel Edge Detection":
        verbose = params.get("verbose", 0)
        return filters.sobel_edge_detection(image_data, verbose=verbose)
    
    elif filter_type == "Adaptive Threshold":
        block_size = params.get("block_size", 11)
        C = params.get("C", 2)
        verbose = params.get("verbose", 0)
        return filters.adaptive_threshold(image_data, block_size=block_size, C=C, verbose=verbose)
    
    elif filter_type == "Laplacian Edge Detection":
        ksize = params.get("kernel_size", 3)
        verbose = params.get("verbose", 0)
        return filters.laplacian_filter(image_data, ksize=ksize, verbose=verbose)
    
    return image_data


def main():
    st.title("🧠 MediFilter: Advanced Medical Imaging Filters")
    st.markdown("""
    <div class="filter-card">
    <h3>Welcome to MediFilter!</h3>
    <p>This application allows you to apply advanced filters to medical images, including:</p>
    <ul>
        <li><strong>DoG (Difference of Gaussian)</strong>: Enhances edges in medical images</li>
        <li><strong>Frangi Filter</strong>: Enhances vessel-like structures</li>
        <li><strong>Sobel Edge Detection</strong>: Highlights edges in the image</li>
        <li><strong>Adaptive Threshold</strong>: Creates binary images with adaptive thresholding</li>
        <li><strong>Laplacian Edge Detection</strong>: Detects edges using second derivatives</li>
    </ul>
    <p>Upload an image file (.jpg, .png), a DICOM file, or a NIfTI file (.nii, .nii.gz) to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for upload and configuration
    with st.sidebar:
        st.header("Input Image")
        
        uploaded_file = st.file_uploader(
            "Upload a medical image file", 
            type=["jpg", "jpeg", "png", "nii", "nii.gz", "dcm"]
        )
        
        st.header("Filter Selection")
        filter_type = st.selectbox(
            "Select filter type",
            [
                "DoG (Difference of Gaussian)", 
                "Frangi (Vessel Enhancement)",
                "Sobel Edge Detection",
                "Adaptive Threshold",
                "Laplacian Edge Detection"
            ]
        )
        
        # Filter parameters based on filter type
        st.header("Filter Parameters")
        
        params = {}
        if filter_type == "DoG (Difference of Gaussian)":
            params["fwhm"] = st.slider("FWHM (Full Width at Half Maximum)", 1, 10, 3, 1)
            
        elif filter_type == "Frangi (Vessel Enhancement)":
            params["scale_min"] = st.slider("Min Scale", 1, 5, 1, 1)
            params["scale_max"] = st.slider("Max Scale", 5, 20, 10, 1)
            params["scale_step"] = st.slider("Scale Step", 1, 5, 2, 1)
            params["beta"] = st.slider("Beta", 0.1, 2.0, 0.5, 0.1)
            params["black_ridges"] = st.checkbox("Detect Black Ridges", True)
            params["use_itk"] = st.checkbox("Use ITK (faster)", True)
            
        elif filter_type == "Sobel Edge Detection":
            st.markdown("No parameters needed for Sobel filter.")
            
        elif filter_type == "Adaptive Threshold":
            params["block_size"] = st.slider("Block Size (must be odd)", 3, 99, 11, 2)
            # Make sure block size is odd
            if params["block_size"] % 2 == 0:
                params["block_size"] += 1
            params["C"] = st.slider("C Value", -10, 10, 2, 1)
            
        elif filter_type == "Laplacian Edge Detection":
            params["kernel_size"] = st.slider("Kernel Size", 1, 7, 3, 2)
            
        process_button = st.button("Process Image")
        
        st.header("About")
        st.markdown("""
        <div style="font-size: 0.85em;">
        <p>MediFilter is a medical imaging application that provides various filters
        for enhancing structures and features in medical images.</p>
        <p>Created with Streamlit, Nibabel, and SciPy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if uploaded_file is not None:
        # Load the image
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        try:
            if file_ext in ['.nii', '.gz']:
                # Handle NIfTI files
                # Create a temporary file that stays open while we use it
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                temp_file.write(uploaded_file.getvalue())
                temp_file.flush()
                temp_file.close()
                
                try:
                    img = nib.load(temp_file.name)
                    # Extract data immediately to avoid file access issues
                    original_data = img.get_fdata().copy()  # Create an explicit copy of the data
                    # Create a new NIfTI image with the copied data to avoid file access issues
                    image_data = nib.Nifti1Image(original_data, img.affine, img.header)
                    is_nifti = True
                finally:
                    # Clean up the temp file
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                
            elif file_ext in ['.jpg', '.jpeg', '.png']:
                # Handle standard image files
                image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
                image_data = np.array(image)
                original_data = image_data.copy()
                is_nifti = False
                
            elif file_ext == '.dcm':
                st.warning("DICOM support is limited in this version. Consider converting to NIfTI format for better results.")
                # Basic DICOM support would require pydicom library
                # For simplicity, this example doesn't fully implement DICOM support
                st.stop()
            
            # Display original image
            st.markdown("<div class='results-section'>", unsafe_allow_html=True)
            st.subheader("Original Image")
            
            if is_nifti:
                # For NIfTI, show middle slice
                fig = plot_slice(original_data, "Original Image")
                st.pyplot(fig)
                st.text(f"NIfTI Image Shape: {original_data.shape}")
            else:
                # For regular images
                fig = plot_slice(original_data, "Original Image")
                st.pyplot(fig)
                st.text(f"Image Shape: {original_data.shape}")
            
            # Process the image when button is clicked
            if process_button:
                with st.spinner(f"Applying {filter_type} filter..."):
                    # Apply selected filter
                    if is_nifti and filter_type == "DoG (Difference of Gaussian)":
                        # For NIfTI files and DoG filter, we need to use the NIfTI image object
                        filtered_image = apply_filter(image_data, filter_type, params)
                    else:
                        # For other cases, use the numpy array
                        filtered_image = apply_filter(original_data, filter_type, params)
                    
                    if isinstance(filtered_image, nib.Nifti1Image):
                        filtered_data = filtered_image.get_fdata()
                    else:
                        filtered_data = filtered_image
                    
                    # Display filtered image
                    st.subheader(f"Filtered Image: {filter_type}")
                    
                    # Determine appropriate colormap
                    if filter_type == "DoG (Difference of Gaussian)":
                        cmap = 'gray'
                    elif filter_type == "Frangi (Vessel Enhancement)":
                        cmap = 'inferno'
                    elif filter_type in ["Sobel Edge Detection", "Laplacian Edge Detection"]:
                        cmap = 'viridis'
                    elif filter_type == "Adaptive Threshold":
                        cmap = 'gray'
                    else:
                        cmap = 'gray'
                    
                    fig = plot_slice(filtered_data, f"Result: {filter_type}", colormap=cmap)
                    st.pyplot(fig)
                    
                    # Download options
                    if is_nifti:
                        # For NIfTI, offer to save as NIfTI
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
                            # Save filtered image as NIfTI
                            if not isinstance(filtered_image, nib.Nifti1Image):
                                # If it's a numpy array, create a NIfTI image with original affine
                                filtered_image = nib.Nifti1Image(filtered_data, img.affine, img.header)
                            
                            nib.save(filtered_image, tmp.name)
                            
                            with open(tmp.name, 'rb') as f:
                                st.download_button(
                                    label="Download Filtered NIfTI",
                                    data=f.read(),
                                    file_name=f"filtered_{filter_type.replace(' ', '_')}_{Path(uploaded_file.name).stem}.nii.gz",
                                    mime="application/octet-stream"
                                )
                            
                            # Don't try to delete the file immediately
                            # Let the OS clean it up when it's no longer in use
                            # Safely schedule deletion for next Streamlit run
                            try:
                                import atexit
                                atexit.register(lambda file=tmp.name: os.unlink(file) if os.path.exists(file) else None)
                            except Exception:
                                # If deletion fails, it's not critical
                                pass
                    
                    else:
                        # For regular images, offer to save as PNG
                        # Normalize and convert to uint8
                        filtered_display = normalize_image_for_display(filtered_data) * 255
                        filtered_display = filtered_display.astype(np.uint8)
                        
                        # Create PIL image
                        result_img = Image.fromarray(filtered_display)
                        
                        # Save to bytes buffer
                        buf = io.BytesIO()
                        result_img.save(buf, format="PNG")
                        
                        # Add download button
                        st.download_button(
                            label="Download Filtered Image",
                            data=buf.getvalue(),
                            file_name=f"filtered_{filter_type.replace(' ', '_')}_{Path(uploaded_file.name).stem}.png",
                            mime="image/png"
                        )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.exception(e)
    
    else:
        # Display sample images when no file is uploaded
        st.markdown("<div class='results-section'>", unsafe_allow_html=True)
        st.subheader("Sample Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Original Brain MRI")
            st.image("https://i.imgur.com/8r2SMnY.png", use_column_width=True)
        
        with col2:
            st.markdown("##### After DoG Filter")
            st.image("https://i.imgur.com/JwcuV5x.png", use_column_width=True)
        
        st.markdown("""
        <p style="text-align: center; font-style: italic;">
        Upload your own medical images to apply various filters and enhance specific features.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
