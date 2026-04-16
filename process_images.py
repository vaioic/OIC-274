from pathlib import Path
from bioio import BioImage
from matplotlib import pyplot as plt
import skimage
from math import sqrt
import numpy as np
import xarray as xr
import pandas as pd
import scipy.ndimage as ndi

def read_image(filepath):

    if not (isinstance(filepath, str) or isinstance(filepath, Path)):
        raise ValueError("The filepath argument must be a string.")
    
    img = BioImage(filepath)

    frame = img.get_image_data("YXC", T=0, Z=0)
    # print(img.dims)
    # print(frame.shape)
    # plt.imshow(frame)               
    # plt.show()
    return frame

def process_files_in_dir(input_dir, output_dir, mask_dir=None, file_ext="czi"):

    # Validate the inputs
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    elif isinstance(input_dir, Path):
        pass
    else:
        raise TypeError(f"Expected input_dir to be a str or Path. Instead it is a {type(input_dir)}.")
    
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    elif isinstance(output_dir, Path):
        pass
    else:
        raise TypeError(f"Expected output_dir to be a str or Path. Instead it is a {type(output_dir)}.")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    elif not output_dir.is_dir():
        raise ValueError(f"The output path {output_dir} points to a file, not a directory.")
    
    if isinstance(mask_dir, str):
        mask_dir = Path(mask_dir)
    elif isinstance(mask_dir, Path):
        pass
    else:
        raise TypeError(f"Expected mask_dir to be a str or Path. Instead it is a {type(mask_dir)}.")

    # Check if the output CSV-file exists and is locked by another program 
    # (like Excel)
    output_csv = output_dir / "results.csv"
    if (output_csv).exists():
        try:
            # We try to open for appending; if locked, this fails instantly
            with open(output_csv, 'a'):
                pass 
        except PermissionError:
            raise PermissionError(f"ERROR: The file '{output_csv}' is currently open in Excel or another program.")

    if mask_dir is None:
        # Get files from folder
        files = input_dir.glob(f"*.{file_ext}")

        all_results = []

        for f in files:
            print(f)
            ds = analyze_image(f, output_dir)
            all_results.append(ds)
        
        all_ds = xr.concat(all_results, dim="id", join="outer")
        all_ds.to_netcdf(output_dir / "results.nc")

    else:
        # Get mask file names
        mask_files = mask_dir.glob(f"*_labels.tif")

        all_results = []

        for m in mask_files:

            f = m.name.replace('_labels.tif', f".{file_ext}")

            print(f"Processing file {input_dir / f} using mask {m}.")

            ds = analyze_image(input_dir / f, output_dir, mask_path=m)
            all_results.append(ds)
        
        all_ds = xr.concat(all_results, dim="id", join="outer")
        all_ds.to_netcdf(output_dir / "results.nc")

    # Save as CSV
    df = all_ds.to_dataframe()

    df = df.rename(columns={
        'feret_diameter_max': 'spot_diameter'
    })

    col_order = ['image', 'cell_label', 'spot_label', 'spot_diameter', 'intensity_mean']
    
    df.to_csv(output_dir / "results.csv", columns=col_order, index=False)

    # Calculate summary statistics by image
    cell_count = df.groupby('image')['cell_label'].nunique()
    cell_count_xr = cell_count.to_xarray().rename("cell_count")
    
    # Calculate summary statistics - change to number of spots per cell, number of cells etc.
    spots_per_cell = all_ds.spot_label.groupby(["image", "cell_label"]).count().rename("spot_count")
    spot_intensity_per_cell = all_ds.intensity_mean.groupby(["image", "cell_label"]).mean().rename("avg_intensity")

    mean_spots_per_cell = spots_per_cell.groupby("image").mean()

    mean_spot_intensity_per_cell = spot_intensity_per_cell.groupby("image").mean()

    summary_ds = xr.merge([cell_count_xr, mean_spots_per_cell, mean_spot_intensity_per_cell])
    summary_df = summary_ds.to_dataframe()
    # df = df.rename(columns={
    #     'feret_diameter_max': 'spot_diameter'
    # })
    summary_df.to_csv(output_dir / "summary.csv", index=True)

def analyze_image(filepath, outputpath, mask_path=None, segment_only=False):
    """
    Analyze a single image, identifying individual nuclei and labeling each 
    spot.

    Parameters
    ----------
    filepath : Path
        Path to the image file
    outputpath : Path
        Path to save the labeled image file

    Returns
    -------
    ds : xarray.Dataset
        A dataset containing the measured properties of all spots in individual cells in the image. Properties include the mean intensity and the estimated diameter (max feret) of each spot.
    """

    if isinstance(filepath, str):
        filepath = Path(filepath)
    elif isinstance(filepath, Path):
        pass
    else:
        raise ValueError('Expected filepath to be a str or Path.')
    
    if isinstance(outputpath, str):
        outputpath = Path(outputpath)
    elif isinstance(outputpath, Path):
        pass
    else:
        raise ValueError('Expected outputpath to be a str or Path.')
    
    if not outputpath.exists():
        outputpath.mkdir(parents=True)

    # Image channels are C=0 (puncta), C=1 nucleus
    img = read_image(filepath)

    # Use masks if provided
    if mask_path is not None:
        cell_labels = skimage.io.imread(mask_path)

        # Remove labels that intersect with image border
        cell_labels = skimage.segmentation.clear_border(cell_labels)        
        
    else:
        cell_labels, raw_labels = segment_nuclei(img[..., 1])

    # Primarily for debugging
    if segment_only:
        overlay = skimage.segmentation.mark_boundaries(
            skimage.exposure.rescale_intensity(img[..., 1], out_range=(0.0, 1.0)), 
            cell_labels, 
            color=(1, 1, 0), 
            mode='thick')
        plt.imshow(overlay)
        plt.show()
        exit()

    spot_labels = segment_spots(img[..., 0], cell_labels)

    cell_props = skimage.measure.regionprops(cell_labels)

    results = []

    for cell in cell_props:

        curr_spot_labels = np.zeros_like(spot_labels)

        curr_spot_labels[cell_labels==cell.label] = spot_labels[cell_labels==cell.label]

        # Count number of spots
        spot_props = skimage.measure.regionprops_table(curr_spot_labels, img[..., 0], properties=('label', 'intensity_mean', 'feret_diameter_max'))
        
        # Build an xarray
        num_spots = len(spot_props['label'])
        curr_ds = xr.Dataset(
            data_vars={
                key: (["id"], val)
                for key, val in spot_props.items() if key != 'label'
            },
            coords={
                'spot_label': (["id"], spot_props['label']),
                'cell_label': (["id"], np.repeat(cell.label, num_spots)),
                'image': (["id"], [str(filepath.stem)] * num_spots)
            }
        )

        results.append(curr_ds)

    ds = xr.concat(results, dim="id", join="outer")

    rgb_image = generate_rgb_image(img)

    # Overlay cell outlines
    # overlay = skimage.color.label2rgb(cell_labels, rgb_image, bg_label=0, kind='overlay', image_alpha=0.8)
    overlay = skimage.segmentation.mark_boundaries(rgb_image, cell_labels, color=(1, 1, 0), mode='thick')

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(overlay)

    # 2. Add cell numbers
    for prop in cell_props:
        y, x = prop.centroid
        ax[0].text(x, y, str(prop.label), color='yellow', fontsize=9, fontweight='normal')

    # 3. Add spots as distinct markers
    # We use a scatter plot so they stand out against the background
    spot_props = skimage.measure.regionprops(spot_labels)
    if spot_props:
        sy, sx = zip(*[p.centroid for p in spot_props])
        ax[0].scatter(sx, sy, s=1, marker='.', label='Spots', color='cyan', linewidths=0.5)

    # Normalize the spot image
    spot_ch = (img[..., 0]).copy()
    spot_ch = (spot_ch - np.min(spot_ch))/(np.max(spot_ch) - np.min(spot_ch))

    spot_ch = 3 * spot_ch
    spot_ch = np.clip(spot_ch, 0, 1)

    spot_ch_overlay = skimage.segmentation.mark_boundaries(spot_ch, cell_labels, color=(1, 1, 0), mode='thick')

    ax[1].imshow(spot_ch_overlay)
    # if spot_props:
    #     sy, sx = zip(*[p.centroid for p in spot_props])
    #     ax[1].scatter(sx, sy, s=5, marker='+', label='Spots', facecolors='none', edgecolors='cyan', linewidths=0.5)

    plt.savefig(outputpath / (filepath.stem + ".png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    # Also save the spot image on its own

    if mask_path is None:
        # Save the raw labels if a pre-made mask was not used
        raw_label_matrix_16 = raw_labels.astype(np.uint16)

        # Save as a TIFF
        skimage.io.imsave(outputpath / (filepath.stem + "_labels.tif"), raw_label_matrix_16, check_contrast=False)

    return ds

def segment_cells_cp(image_dir, output_dir, cell_diameter=125):
    """
    Generate segmentation masks for the nuclei using Cellpose.

    This function uses the Cellpose 'nuclei' model to help segment nuclei labeled with DAPI. The inferred labels will be saved as a TIFF-file in the output_dir.
    
    Parameters
    ----------
    image_dir : str or Path
        Path to the directory of images
    output_dir : str or Path
        Path to the output directory to save masks
    cell_diameter : int, optional
        Estimated cell diameter used by Cellpose, by default 125

    Raises
    ------
    ValueError
        The image_dir must be a str or Path
    ValueError
        The output_dir must be a str or Path
    """

    # Validate the input directory
    if isinstance(image_dir, str):
        image_dir = Path(image_dir)
    elif isinstance(image_dir, Path):
        pass
    else:
        raise ValueError(f"Expected image_dir to be a str or Path. Instead it is {type(image_dir)}.")
    
    # Validate the output directory
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    elif isinstance(output_dir, Path):
        pass
    else:
        raise ValueError(f"Expected output_dir to be a str or Path. Instead it is {type(output_dir)}.")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Run cellpose    
    from cellpose import models

    model = models.CellposeModel(gpu=False, model_type='nuclei')

    files = image_dir.glob('*.czi')

    for f in files:
        img = read_image(f)

        img_nucl = img[..., 1]
        
        # Normalize the intensity of the image
        img_norm = skimage.exposure.rescale_intensity(img_nucl, out_range=(0.0, 1.0))    

        mask, _, _ = model.eval(img_norm, diameter=cell_diameter)
        
        skimage.io.imsave(output_dir / (f.stem + "_labels.tif"), mask.astype('uint16'))



def generate_rgb_image(img):

    # Create an overlay image
    img = img.astype('float32')
    img[..., 1] = (img[..., 1] - np.min(img[..., 1]))/(np.max(img[..., 1]) - np.min(img[..., 1]))
    img[..., 0] = (img[..., 0] - np.min(img[..., 0]))/(np.max(img[..., 0]) - np.min(img[..., 0]))

    img[..., 0] = 3 * img[..., 0]
    img[..., 0] = np.clip(img[..., 0], 0, 1)

    rgb_image = np.zeros((img.shape[0], img.shape[1], 3))
    # alpha = 0.4
    rgb_image[..., 0] = img[..., 0]
    # rgb_image[..., 1] = alpha * img[..., 0]
    rgb_image[..., 2] = img[..., 0] + img[..., 1]

    rgb_image = np.clip(rgb_image, 0, 1)

    return rgb_image
   

def segment_nuclei(img):
    """
    Segment nuclei using standard Otsu's thresholding and morphological 
    operations to clean the mask.

    Parameters
    ----------
    img : ndarray
        A single-channel (grayscale) image of the nuclear marker. The image does not need to be normalized.

    Returns
    -------
    labels : ndarray(dtype=int)
        An ndarray with the same shape as the input image, where each pixel's value represents its assigned label.
    """

    filtered_img = skimage.filters.median(img, skimage.morphology.disk(5))
    filtered_img = skimage.filters.gaussian(img, 2)

    filtered_img = (filtered_img - np.min(filtered_img))/(np.max(filtered_img) - np.min(filtered_img))

    threshold = skimage.filters.threshold_otsu(filtered_img)

    mask = filtered_img > (0.5 * threshold)

    mask = skimage.morphology.isotropic_opening(mask, 15)

    mask = skimage.morphology.remove_small_holes(mask, max_size=10000)
    mask = skimage.morphology.remove_small_objects(mask, max_size=1000)

    mask = skimage.morphology.isotropic_closing(mask, 5)

    mask = skimage.segmentation.clear_border(mask)

    # Watershed
    dd = ndi.distance_transform_edt(mask)

    #print(np.max(dd))
    h = 0.2 * np.max(dd)
    h_filtered = skimage.morphology.h_minima(-dd, h)
    
    markers, _ = ndi.label(h_filtered)
    labels = skimage.segmentation.watershed(-dd, markers, mask=mask)

    if not np.any(labels > 0):
        raise ValueError("No objects were found.")
    
    # # Filter objects by size for images with > 10 objects
    # props = skimage.measure.regionprops(labels)
    # areas = np.array([p.area for p in props])
    # label_ids = np.array([p.label for p in props])
    # if len(areas) > 10:
    #     # 3. Calculate IQR (Interquartile Range) for Area
    #     # This identifies the "typical" size range of your objects.
    #     q1, q3 = np.percentile(areas, [25, 75])
    #     iqr = q3 - q1
        
    #     # Define bounds (standard multiplier is 1.5)
    #     # Lower bound prevents noise; upper bound prevents merged "clumps"
    #     lower_bound = q1 - (1.5 * iqr)
    #     upper_bound = q3 + (1.5 * iqr)
        
    #     # 4. Filter labels based on these statistical bounds
    #     # We also ensure the area is at least a few pixels to catch tiny noise
    #     keep_indices = (areas >= max(5, lower_bound)) & (areas <= upper_bound)
    #     valid_labels = label_ids[keep_indices]
        
    #     # 5. Create the final cleaned label map
    #     # np.isin is the most efficient way to zero out the "bad" labels
    #     mask = np.isin(labels, valid_labels)
    #     cleaned_labels = np.where(mask, labels, 0)
    # else:
    #     cleaned_labels = labels

    cleaned_labels = labels

    return cleaned_labels, labels

def segment_spots(img, cell_labels=None, spot_thresh=0.02):
    """
    Segment spots in images

    This function uses the difference of Gaussians method to identify spots in images.

    Parameters
    ----------
    img : ndarray
        Image of the spot channel.
    cell_labels : ndarray, optional
        Cell labels, by default None. If provided, the function will remove spots that are outside of cells.
    spot_thresh : float, optional
        Threshold of difference to use for spots, by default 0.02. Increasing this value will reduce the number of labelled pixels.

    Returns
    -------
    _type_
        _description_
    """

    filtered_img = skimage.filters.median(img, skimage.morphology.disk(2))

    filtered_img = (filtered_img - np.min(filtered_img))/(np.max(filtered_img) - np.min(filtered_img))

    diff_of_gaussians = skimage.filters.difference_of_gaussians(filtered_img, 2, 8)

    # # Calculate a threshold value
    # global_spot_mean = np.mean(filtered_img)
    # background_mean = np.mean(filtered_img[filtered_img < global_spot_mean])
    # background_std = np.std(filtered_img[filtered_img < global_spot_mean])
    # spot_thresh = background_mean + 3 * background_std
    # bright_regions = filtered_img > spot_thresh

    # plt.imshow(bright_regions)
    # plt.show()
    # # exit()

    spot_mask = diff_of_gaussians > (5 * np.std(diff_of_gaussians))

    # plt.imshow(spot_mask)
    # plt.show()
    # exit()

    # spot_mask = spot_mask & bright_regions


    if cell_labels is not None:
        # Shrink the cell labels a little to avoid spots along the edge
        cell_labels = skimage.morphology.erosion(cell_labels, skimage.morphology.disk(3))

        spot_mask = spot_mask & (cell_labels > 0)

    spot_labels = skimage.measure.label(spot_mask)

    return spot_labels


if __name__ == "__main__":

    # analyze_image(r"D:\Projects\OIC-274 Rahma\data\03042026\96wellplate_63x_03042026_processed-Scene-010-P1-B06.czi", r"D:\Projects\OIC-274 Rahma\processed\2026-04-07", segment_only=True, useTestSeg=True)
    # process_files_in_dir(r"D:\Projects\OIC-274 Rahma\data\03042026", r"D:\Projects\OIC-274 Rahma\processed\2026-04-07b")

    # analyze_image(r"D:\Projects\OIC-274 Rahma\data\03042026\96wellplate_63x_03042026_processed-Scene-033-P2-C09.czi", r"D:\Projects\OIC-274 Rahma\processed\2026-04-07", segment_only=True, useTestSeg=True)

    # segment_cells_cp(r"D:\Projects\OIC-274 Rahma\data\03042026", r"D:\Projects\OIC-274 Rahma\processed\2026-04-07b\masks")

    process_files_in_dir(r"D:\Projects\OIC-274 Rahma\data\03042026", r"D:\Projects\OIC-274 Rahma\processed\2026-04-15", mask_dir=r"D:\Projects\OIC-274 Rahma\processed\2026-04-09b\masks")