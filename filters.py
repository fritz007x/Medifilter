#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.filters import frangi, hessian
import cv2
import warnings

# Import SimpleITK for the optimized Frangi filter
try:
    import SimpleITK as sitk
    HAVE_SITK = True
except ImportError:
    HAVE_SITK = False
    warnings.warn("SimpleITK not found. Using slower scikit-image implementation for Frangi filter.")

def frangi_filter(img, scale_range=(1, 10), scale_step=2, beta=0.5, black_ridges=True, verbose=0, use_itk=True):
    """
    Apply Frangi vesselness filter to enhance vessel-like structures in images.
    
    Parameters
    ----------
    img : Niimg-like object
        Image to apply the Frangi filter to.
    scale_range : tuple, optional
        The range of sigmas used for computing filter response.
    scale_step : float, optional
        Step size between sigmas.
    beta : float, optional
        Frangi correction constant.
    black_ridges : bool, optional
        When True, the filter detects black ridges; when False, it detects white ridges.
    verbose : int, optional
        Controls the amount of verbosity: higher numbers give more messages.
    use_itk : bool, optional
        If True (default), uses the faster SimpleITK implementation when available.
        If False or SimpleITK is not installed, falls back to scikit-image implementation.
        
    Returns
    -------
    nibabel.nifti1.Nifti1Image or numpy array
        Filtered image with enhanced vessel-like structures.
    """
    if verbose > 0:
        print(f"Applying Frangi filter with scale range {scale_range}, beta {beta}")
        print(f"Using {'SimpleITK' if use_itk and HAVE_SITK else 'scikit-image'} implementation")
        if hasattr(img, 'shape'):
            print(f"Input image shape: {img.shape}")
    
    # Get image data and metadata
    if hasattr(img, 'get_fdata'):
        data = img.get_fdata()
        affine = img.affine
        header = img.header
    else:
        # Assume it's a numpy array for non-NIfTI images
        data = np.array(img)
        affine = None
        header = None
    
    # Use SimpleITK implementation if available and requested
    if use_itk and HAVE_SITK:
        # Handle 2D and 3D images differently
        is_3d = (data.ndim == 3)
        
        # Starting and stopping sigmas for the filter
        start_sigma = scale_range[0]
        end_sigma = scale_range[1] 
        sigma_step = scale_step
        
        # Convert numpy array to SimpleITK image
        sitk_img = sitk.GetImageFromArray(data)
        
        # Configure Frangi filter parameters
        frangi_filter = sitk.FrangiVesselness(
            minimumDiameter=start_sigma, 
            maximumDiameter=end_sigma,
            # Convert step size to number of steps
            numberOfSteps=int((end_sigma - start_sigma) / sigma_step) + 1,
            # Dimensionality, important for correct computation
            dimension=3 if is_3d else 2,
            # Other parameters
            brightObjects=not black_ridges,
            # beta1/2 control sensitivity to blob vs tube shapes
            alpha=0.5, beta=beta, gamma=5.0
        )
        
        try:
            # Apply the filter
            filtered_sitk = frangi_filter.Execute(sitk_img)
            
            # Convert back to numpy array
            filtered_data = sitk.GetArrayFromImage(filtered_sitk)
            
            # Normalize result to 0-1 range
            if np.max(filtered_data) > 0:
                filtered_data = filtered_data / np.max(filtered_data)
                
        except Exception as e:
            if verbose > 0:
                print(f"SimpleITK Frangi filter failed: {str(e)}")
                print("Falling back to scikit-image implementation")
            # Fall back to scikit-image implementation
            filtered_data = frangi(data, 
                                scale_range=scale_range, 
                                scale_step=scale_step, 
                                beta=beta, 
                                black_ridges=black_ridges)
    else:
        # Use scikit-image implementation
        filtered_data = frangi(data, 
                            scale_range=scale_range, 
                            scale_step=scale_step, 
                            beta=beta, 
                            black_ridges=black_ridges)
    
    # Create output
    if affine is not None and header is not None:
        # Create NIfTI image with original header and affine
        out_img = nib.Nifti1Image(filtered_data, affine, header)
        return out_img
    else:
        # Return numpy array for non-NIfTI images
        return filtered_data

def sobel_edge_detection(img, verbose=0):
    """
    Apply Sobel edge detection filter.
    
    Parameters
    ----------
    img : Niimg-like object or numpy array
        Image to apply the Sobel filter to.
    verbose : int, optional
        Controls the amount of verbosity: higher numbers give more messages.
        
    Returns
    -------
    nibabel.nifti1.Nifti1Image or numpy array
        Filtered image with enhanced edges.
    """
    if verbose > 0:
        print("Applying Sobel edge detection")
    
    # Get image data
    if hasattr(img, 'get_fdata'):
        data = img.get_fdata()
        affine = img.affine
        header = img.header
    else:
        # Assume it's a numpy array for non-NIfTI images
        data = np.array(img)
        affine = None
        header = None
    
    # Apply Sobel filter
    dx = ndimage.sobel(data, axis=0)
    dy = ndimage.sobel(data, axis=1)
    
    if data.ndim == 3:
        dz = ndimage.sobel(data, axis=2)
        magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
    else:
        magnitude = np.sqrt(dx**2 + dy**2)
    
    # Normalize to 0-1 range
    if np.max(magnitude) > 0:
        magnitude = magnitude / np.max(magnitude)
    
    if affine is not None and header is not None:
        # Create NIfTI image with original header and affine
        out_img = nib.Nifti1Image(magnitude, affine, header)
        return out_img
    else:
        # Return numpy array for non-NIfTI images
        return magnitude

def adaptive_threshold(img, block_size=11, C=2, verbose=0):
    """
    Apply adaptive thresholding to an image.
    
    Parameters
    ----------
    img : Niimg-like object or numpy array
        Image to apply adaptive thresholding to.
    block_size : int, optional
        Size of a pixel neighborhood used to calculate threshold value.
    C : int, optional
        Constant subtracted from the mean or weighted mean.
    verbose : int, optional
        Controls the amount of verbosity: higher numbers give more messages.
        
    Returns
    -------
    nibabel.nifti1.Nifti1Image or numpy array
        Thresholded binary image.
    """
    if verbose > 0:
        print(f"Applying adaptive threshold with block size {block_size}, C {C}")
    
    # Get image data
    if hasattr(img, 'get_fdata'):
        data = img.get_fdata()
        affine = img.affine
        header = img.header
    else:
        # Assume it's a numpy array for non-NIfTI images
        data = np.array(img)
        affine = None
        header = None
    
    # Convert to uint8 for OpenCV
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    data_uint8 = data_norm.astype(np.uint8)
    
    # Apply threshold
    if data.ndim == 2:
        thresh = cv2.adaptiveThreshold(
            data_uint8, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            block_size, 
            C
        )
    else:
        # For 3D or more dimensions, process slice by slice
        thresh = np.zeros_like(data_uint8)
        for i in range(data.shape[2] if data.ndim >= 3 else 1):
            if data.ndim >= 3:
                thresh[:,:,i] = cv2.adaptiveThreshold(
                    data_uint8[:,:,i], 
                    255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 
                    block_size, 
                    C
                )
    
    # Convert back to binary (0 and 1)
    thresh = thresh / 255
    
    if affine is not None and header is not None:
        # Create NIfTI image with original header and affine
        out_img = nib.Nifti1Image(thresh, affine, header)
        return out_img
    else:
        # Return numpy array for non-NIfTI images
        return thresh

def laplacian_filter(img, ksize=3, verbose=0):
    """
    Apply Laplacian filter for edge detection.
    
    Parameters
    ----------
    img : Niimg-like object or numpy array
        Image to apply the Laplacian filter to.
    ksize : int, optional
        Aperture size used to compute the second-derivative filters.
    verbose : int, optional
        Controls the amount of verbosity: higher numbers give more messages.
        
    Returns
    -------
    nibabel.nifti1.Nifti1Image or numpy array
        Filtered image with enhanced edges.
    """
    if verbose > 0:
        print(f"Applying Laplacian filter with kernel size {ksize}")
    
    # Get image data
    if hasattr(img, 'get_fdata'):
        data = img.get_fdata()
        affine = img.affine
        header = img.header
    else:
        # Assume it's a numpy array for non-NIfTI images
        data = np.array(img)
        affine = None
        header = None
    
    # Apply Laplacian filter
    laplacian = ndimage.laplace(data)
    
    # Normalize to 0-1 range
    laplacian_norm = (laplacian - np.min(laplacian)) / (np.max(laplacian) - np.min(laplacian) + 1e-9)
    
    if affine is not None and header is not None:
        # Create NIfTI image with original header and affine
        out_img = nib.Nifti1Image(laplacian_norm, affine, header)
        return out_img
    else:
        # Return numpy array for non-NIfTI images
        return laplacian_norm
