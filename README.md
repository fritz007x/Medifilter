# Medifilter

This is a web app developed for the course CAI2840C-2253-7384 Introduction to Computer Vision. It is a medical imaging application that incorporates:
* Visualization of both original and filtered images
* Download options for processed images

Key Features:
* Difference of Gaussian (DoG) filter from the original dog.py
* Frangi filter for vessel enhancement (with SimpleITK optimization)
* Sobel Edge Detection: Highlights edges based on intensity gradients
* Adaptive Threshold: Creates binary images with adaptive thresholding
* Laplacian Edge Detection: Detects edges using second derivatives
* Specialized for medical imaging needs
* Parameters can be adjusted for each filter type
* Handles 2D and 3D medical data (showing appropriate slices for 3D)
* Download processed results in appropriate formats

## Supported File Formats
* Standard images (.jpg, .png)
* Medical imaging formats (.nii, .nii.gz)
* Basic DICOM support (.dcm)

## Installation

### Prerequisites

- Python 3.7+ installed
- Pip package manager

### Setup

1. **Clone or download this repository**

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

   If you encounter issues with specific packages, you can try installing them individually:
   ```
   pip install streamlit
   pip install nibabel
   pip install scipy
   pip install numpy
   pip install scikit-image
   pip install opencv-python
   pip install matplotlib
   pip install pillow
   pip install SimpleITK
   ```

   Note: On Windows, you might need to install a C++ build environment for some packages. The simplest solution is to install a pre-compiled wheel:
   ```
   pip install --only-binary=scipy,numpy scipy numpy
   ```

3. **Verify installation:**
   After installation, you can check if the packages are correctly installed:
   ```
   python -c "import streamlit, nibabel, scipy, numpy, cv2, matplotlib; print('All required packages are installed!')"
   ```

## Running the Application

Launch the application with:
```
streamlit run app.py
```

This will start a local web server and automatically open the MediFilter application in your default web browser.

## Using MediFilter

1. **Upload an Image:**
   - Use the file uploader in the sidebar to select a medical image file
   - The application supports .jpg, .png, .nii, .nii.gz, and .dcm formats

2. **Select a Filter:**
   - Choose from the dropdown menu in the sidebar
   - Each filter has different parameters you can adjust

3. **Adjust Parameters:**
   - Modify the filter parameters based on your needs
   - Different filters have different adjustable parameters

4. **Process the Image:**
   - Click the "Process Image" button to apply the selected filter
   - View the processed image in the main panel

5. **Download Results:**
   - Use the download button to save the processed image
   - File formats are preserved (NIfTI will be saved as .nii.gz, standard images as .png)
