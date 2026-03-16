from pathlib import Path
from bioio import BioImage
from matplotlib import pyplot as plt
import skimage
from math import sqrt
import numpy as np
import xarray as xr
import pandas as pd

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

def process_files_in_dir(inputpath, outputpath, file_ext="czi"):

    # Validate the inputs
    ip = Path(inputpath)

    op = Path(outputpath)

    # Check if the output CSV-file exists and is locked by another program 
    # (like Excel)
    output_csv = op / "results.csv"
    if (output_csv).exists():
        try:
            # We try to open for appending; if locked, this fails instantly
            with open(output_csv, 'a'):
                pass 
        except PermissionError:
            raise PermissionError(f"ERROR: The file '{output_csv}' is currently open in Excel or another program.")

    if not op.exists():
        op.mkdir()
    
    if not op.is_dir():
        raise ValueError(f"The output path {outputpath} points to a file, not a directory.")

    files = ip.glob(f"*.{file_ext}")

    all_results = []

    for f in files:
        print(f)
        ds = analyze_image(f, op)
        all_results.append(ds)
    
    all_ds = xr.concat(all_results, dim="id", join="outer")
    all_ds.to_netcdf(op / "results.nc")

    # Save as CSV
    df = all_ds.to_dataframe()

    df = df.rename(columns={
        'feret_diameter_max': 'spot_diameter'
    })

    col_order = ['image', 'cell_label', 'spot_label', 'spot_diameter', 'intensity_mean']
    
    df.to_csv(op / "results.csv", columns=col_order, index=False)

    # Calculate summary statistics
    counts = ds.spot_label.groupby(["image", "cell_label"]).count().rename("spot_count")
    means = ds.intensity_mean.groupby(["image", "cell_label"]).mean().rename("avg_intensity")

    summary_ds = xr.merge([counts, means])
    summary_df = summary_ds.to_dataframe()
    summary_df.to_csv(op / "summary.csv", index=True)

def analyze_image(filepath, outputpath):
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

    # Image channels are C=0 (puncta), C=1 nucleus
    img = read_image(filepath)

    cell_labels = segment_nuclei(img[..., 1])
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
                key: (['id'], val)
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

    # Create an overlay image
    rgb_image = np.zeros((img.shape[0], img.shape[1], 3))
    img = img.astype('float32')
    img[..., 1] = (img[..., 1] - np.min(img[..., 1]))/(np.max(img[..., 1]) - np.min(img[..., 1]))
    img[..., 0] = (img[..., 0] - np.min(img[..., 0]))/(np.max(img[..., 0]) - np.min(img[..., 0]))

    # print(img.dtype)
    # print(rgb_image.dtype)

    alpha = 0.8
    rgb_image[..., 0] = alpha * img[..., 1] + (1 - alpha) * img[..., 0]
    rgb_image[..., 1] = alpha * img[..., 1]
    rgb_image[..., 2] = alpha * img[..., 1] + (1 - alpha) * img[..., 0]

    # plt.imshow(rgb_image)
    # plt.show()

    overlay = skimage.color.label2rgb(cell_labels, rgb_image, bg_label=0, kind='overlay', image_alpha=0.8)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)

    # 2. Add cell numbers
    for prop in cell_props:
        y, x = prop.centroid
        ax.text(x, y, str(prop.label), color='yellow', fontsize=9, fontweight='bold')

    # 3. Add spots as distinct markers
    # We use a scatter plot so they stand out against the background
    spot_props = skimage.measure.regionprops(spot_labels)
    if spot_props:
        sy, sx = zip(*[p.centroid for p in spot_props])
        ax.scatter(sx, sy, s=3, c='cyan', marker='+', label='Spots')

    plt.legend()
    plt.savefig(outputpath / (filepath.stem + ".png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    return ds

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

    mask = skimage.morphology.opening(mask, skimage.morphology.disk(5))

    mask = skimage.morphology.remove_small_holes(mask, max_size=3000)
    mask = skimage.morphology.remove_small_objects(mask, max_size=500)

    mask = skimage.segmentation.clear_border(mask)

    labels = skimage.measure.label(mask)

    return labels

def segment_spots(img, cell_labels=None):

    filtered_img = skimage.filters.median(img, skimage.morphology.disk(2))

    filtered_img = (filtered_img - np.min(filtered_img))/(np.max(filtered_img) - np.min(filtered_img))

    diff_of_gaussians = skimage.filters.difference_of_gaussians(filtered_img, 2, 8)

    spot_mask = diff_of_gaussians > 0.02

    if cell_labels is not None:
        spot_mask = spot_mask & (cell_labels > 0)

    spot_labels = skimage.measure.label(spot_mask)

    return spot_labels

    # plt.subplot(1, 2, 1)
    # plt.imshow(diff_of_gaussians)
    # plt.subplot(1, 2, 2)
    # plt.imshow(spot_mask)
    # plt.show()

    # print(np.max(filtered_img), np.min(filtered_img))

    # blobs = skimage.feature.blob_dog(filtered_img, min_sigma=0.1, max_sigma=10, threshold=0.01)

    # blobs[:, 2] = blobs[:, 2] * sqrt(2)

    # print("Plotting")

    # fig, ax = plt.subplots(figsize=(9, 9))
    # ax.imshow(img)
    # ax.set_title("Detected Spots (LoG)")

    # for blob in blobs:
    #     y, x, r = blob
    #     # Add a yellow circle for each detected spot
    #     c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
    #     ax.add_patch(c)

    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # print(f"Total spots detected: {len(blobs)}")

    return spot_mask

if __name__ == "__main__":

    # analyze_image(r"D:\Projects\OIC-274 Rahma\data\03042026\96wellplate_63x_03042026_processed-Scene-001-P1-B03.czi")
    process_files_in_dir(r"D:\Projects\OIC-274 Rahma\data\03042026", r"D:\Projects\OIC-274 Rahma\processed\2026-03-16")


