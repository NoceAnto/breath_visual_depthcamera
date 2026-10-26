from scipy.signal import butter, filtfilt
from ultralytics import YOLO # skeleton extraction by YOLO deep learning algorithms
import logging
from ultralytics.utils import LOGGER  # if available in your version
import numpy as np
import cv2


LOGGER.setLevel(logging.ERROR)  # or logging.CRITICAL

def butter_filter(data, cutoff, fs, order=4, ftype='low'):
    """
    data   : input signal (1D array)
    cutoff : cutoff frequency (Hz)
    fs     : sampling frequency (Hz)
    order  : filter order
    ftype  : 'low', 'high', 'bandpass', 'bandstop'
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq if isinstance(cutoff, (int, float)) else [c / nyq for c in cutoff]

    b, a = butter(order, norm_cutoff, btype=ftype)
    filtered = filtfilt(b, a, data)
    return filtered

def colorize_between_values(img_f32, vmin, vmax, cmap=cv2.COLORMAP_TURBO):
    """
    img_f32: single-channel float32/float64 array
    vmin, vmax: float, the value range to map to the colormap
    cmap: any cv2 colormap (e.g., COLORMAP_TURBO, JET, VIRIDIS (OpenCV >= 4.5.2))
    """
    # 1) Handle NaNs if present (optional): set them to vmin so they map to the start color
    img = np.nan_to_num(img_f32, nan=vmin)

    # 2) Clip to the desired window
    img_clipped = np.clip(img, vmin, vmax)

    # Manual scaling (preferred when you want absolute control):
    norm = ((img_clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)

    # 4) Apply a colormap
    color = cv2.applyColorMap(norm, cmap)
    return color


def filter_outlier(matrix, excludeNan=True, excludezero=True,exclude_percentile=True):
    '''
    FUNCTION TO FILTER OUT THE OUTLIER IN A SPECIFIED MATRIX ROI
    :param matrix: matrix of values usually corresponfing to the ROI
    :param excludeNan: True to report a newmatrix vector without the Nan values
    :param excludezero: True to report a newmatrix vector without the zero values
    :param exclude_percentile: True to report a newmatrix vector without the extreme cases values
    :return:
    '''
    if excludeNan:
        newmatrix = matrix[~np.isnan(matrix)]
        if len(newmatrix) == 0:
            corrupted = True
            return newmatrix,corrupted
    # no zeros
    if excludezero:
        newmatrix = newmatrix[newmatrix != 0]
        if len(newmatrix) == 0:
            corrupted = True
            return newmatrix,corrupted

    # Calculate the 0.5 and 99.5 percentiles
    if exclude_percentile:
        percentile_05 = np.percentile(newmatrix, 0.05)
        percentile_995 = np.percentile(newmatrix, 99.5)
        # Filter values based on percentiles
        newmatrix = newmatrix[(newmatrix >= percentile_05) & (newmatrix <= percentile_995)]
        if len(newmatrix) == 0:
            corrupted = True
            return newmatrix,corrupted

    tot_pixels = len(matrix)
    perc_remained_pixels = float(np.size(newmatrix))/float(tot_pixels)

    if perc_remained_pixels<0.8:
        corrupted = True
    else:
        corrupted = False

    return newmatrix, corrupted

def compute_mask(A,B,C,D,depth_image):
    # !! NB. A B C D in the order of connect ex. right shoulder, left shoulder, left hip, right hip
    poly_pts = np.array([A, B, C, D], dtype=np.int32)

    mask = np.zeros_like(depth_image, dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts], 255)
    depth_values = depth_image[mask == 255]

    depth_values, corrupted = filter_outlier(depth_values, excludeNan=True, excludezero=True, exclude_percentile=False)
    if not(corrupted):
        mean_depth = np.mean(depth_values)
        median_depth = np.median(depth_values)
        return mean_depth,median_depth,mask
    else:
        return np.nan,np.nan,mask

def extract_keypoints(model,color_image):
    results = model(color_image)
    high_conf_boxes = []

    all_keypoints = []
    all_keypoints_dict = []
    for r in results:
        boxes = r.boxes
        keypoints = r.keypoints
        # identifico i boxes che hanno confidenza maggiore del 70% per essere sicuri di prendere le persone
        for box in boxes:
            if box.conf >= 0.7:
                high_conf_boxes.append(box)

        # per prendere solo il primo box della lista
        if high_conf_boxes:
            box = high_conf_boxes[0]
            b = box.xyxy[0].to('cpu').detach().numpy().copy()
            c = box.cls

            # trovo i keypoints
            for keypoint in keypoints:
                k = keypoint.xy[0].to('cpu').detach().numpy().copy()
                # round the points to the nearest pixel
                k = np.round(k).astype(int)
                all_keypoints.append(k)
                ### dictionary
                keypoints_dict={
                    "nose": k[0],
                    "left eye": k[1],
                    "right eye": k[2],
                    "left ear": k[3],
                    "right ear": k[4],
                    "left shoulder": k[5],
                    "right shoulder": k[6],
                    "left elbow": k[7],
                    "right elbow": k[8],
                    "left wrist": k[9],
                    "right wrist": k[10],
                    "left hip": k[11],
                    "right hip": k[12],
                    "left knee": k[13],
                    "right knee": k[14],
                    "left ankle": k[15],
                    "right ankle": k[16]
                }
                all_keypoints_dict.append(keypoints_dict)


    return all_keypoints,all_keypoints_dict


# load YOLO pose model
yolomodel = YOLO("yolo11n-pose.pt")

fpath ="depth_video_subject"

file_path_color = fpath+"/color_data.npy"
file_path_depth = fpath+"/depth_data.npy"

print(f"Loading file: {file_path_color}")
colorvideo = np.load(file_path_color);
print(f"Loading file: {file_path_depth}")
depthvideo = np.load(file_path_depth);

[F,height,width] = np.shape(depthvideo);

# extract keypoints from first frame
all_keypoints, all_keypoints_dict = extract_keypoints(yolomodel, colorvideo[0,:,:,:])
subjectskp_dict = all_keypoints_dict[0]

sh1 = subjectskp_dict["right shoulder"]
sh2 = subjectskp_dict["left shoulder"]
hip1 = subjectskp_dict["left hip"]
hip2 = subjectskp_dict["right hip"]

# compute the torso mask
mean_subject_distance, median_subject_distance, mask = compute_mask(sh1, sh2, hip1,hip2,
                                                                    depthvideo[0,:,:])
mask = (depthvideo[0,:,:] >= median_subject_distance - 0.2) & (
         depthvideo[0,:,:] <= median_subject_distance + 0.2)

binary_subject_mask = mask.astype(np.uint8)

subjectdepth = depthvideo.copy()
subjectdepth[:,binary_subject_mask == 0] = 0

subjectdepth_visualvideo=subjectdepth.copy()
for w in range(0, width):
    for h in range(0, height):
        if binary_subject_mask[h, w] != 0:
            subjectdepth_visualvideo[:, h, w] = butter_filter(subjectdepth_visualvideo[:, h, w],
                                                             [0.1,1],fs=15,order=2,ftype="bandpass")

codec = "mp4v"
# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*codec)
output_path_colormap = "video_colormap.mp4"
output_path_color = "video_color.mp4"
output_path_depth = "depthvideo.mp4"

writer_color = cv2.VideoWriter(output_path_color, fourcc, 15, (width, height))
writer_colormap = cv2.VideoWriter(output_path_colormap, fourcc, 15, (width, height))
writer_depth = cv2.VideoWriter(output_path_depth, fourcc, 15, (width, height))

for f in range(0,F):
    colormap = colorize_between_values(subjectdepth_visualvideo[f,:,:], -0.02, 0.02, cmap=cv2.COLORMAP_TURBO)

    colormap[binary_subject_mask == 0,:] = [255,255,255]
    #overlaid = cv2.addWeighted(heatmap_bgr, alpha, frame_bgr, 1 - alpha, 0)
    depth_image = depthvideo[f,:,:]
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=10), cv2.COLORMAP_JET)
    cv2.imshow('colormap',colormap)
    #cv2.imshow('colorvideo + colormap',colorvideo[f,:,:,:]+np.round(0.5*colormap))
    colormap[binary_subject_mask == 0, :] = [0, 0, 0]
    #alpha = 0.2
    #overlaid = cv2.addWeighted(colormap, alpha, colorvideo[f,:,:,:], 1 - alpha, 0)
    #cv2.imshow('overlaid',overlaid)
    cv2.imshow('color',colorvideo[f,:,:,:])
    cv2.imshow('depth',depth_colormap)

    cv2.waitKey(33)

    writer_color.write(colorvideo[f,:,:,:])
    writer_colormap.write(colormap)
    writer_depth.write(depth_colormap)

cv2.destroyAllWindows()
writer_color.release()

print(f"Video saved successfully to {output_path_colormap}")





