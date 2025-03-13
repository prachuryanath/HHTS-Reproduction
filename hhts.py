import cv2
import numpy as np
from typing import List, Optional
from bisect import bisect_right

RGB, HSV, LAB = 1, 2, 4  # Define channel flags

class ColorChannel:
    RGB = 1
    HSV = 2
    LAB = 4

def get_channels(image, color_channels, apply_blur=False):
    channels = []
    blur_size = (3, 3)

    if color_channels & ColorChannel.RGB:
        img = image.copy()
        if apply_blur:
            img = cv2.GaussianBlur(img, blur_size, 0)
        b, g, r = cv2.split(img)
        channels.extend([r, g, b])  # OpenCV uses BGR format by default

    if color_channels & HSV:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if apply_blur:
            img = cv2.GaussianBlur(img, blur_size, 0)
        h, s, v = cv2.split(img)
        channels.extend([h, s, v])

    if color_channels & LAB:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        if apply_blur:
            img = cv2.GaussianBlur(img, blur_size, 0)
        l, a, b = cv2.split(img)
        channels.extend([l, a, b])

    return channels

class split_params:
    def __init__(self, superpixels= None, split_threshold = 0.0,
                 histogram_bins = 16, min_segment_size = 64):
        self.superpixels = superpixels if superpixels is not None else []
        self.split_threshold = split_threshold
        self.histogram_bins = histogram_bins
        self.min_segment_size = min_segment_size

class ChannelInfo:
    def __init__(self, channel, mask, size):
        self.min = int(np.min(channel[mask > 0]))
        self.max = int(np.max(channel[mask > 0]))
        self.width = self.max - self.min

        if self.width < 2:
            self.split_criteria = -1.0
            return

        mean, stddev = cv2.meanStdDev(channel, mask=mask.astype(np.uint8))
        self.split_criteria = (stddev[0, 0] * size * size)

class Label:
    def __init__(self, channels, input_mask, label_size, label_id, split_params):
        self.id = label_id
        self.label_size = label_size
        self.mask = input_mask.copy()
        self.child_min_size = split_params.min_segment_size
        self.label_split_criteria = -1.0
        self.label_split_channel = -1
        self.channel_infos = []
        
        if not self.is_size_splittable():
            return

        for i, channel in enumerate(channels):
            channel_info = ChannelInfo(channel, self.mask, label_size)
            self.channel_infos.append(channel_info)

            if channel_info.split_criteria > self.label_split_criteria:
                self.label_split_criteria = channel_info.split_criteria
                self.label_split_channel = i

    def is_size_splittable(self):
        return self.label_size/2 >= self.child_min_size
    
    def is_splittable(self, split_threshold):
        return self.label_split_criteria > split_threshold and self.is_size_splittable()

    def interrupt_split(self, labels, splittable_labels, split_params):
        # Invalidate current label split channel
        self.channel_infos[self.label_split_channel].split_criteria = -1.0

        # Find new best split channel
        self.label_split_criteria = -1.0
        self.label_split_channel = -1
        for i, channel_info in enumerate(self.channel_infos):
            if channel_info.split_criteria > self.label_split_criteria:
                self.label_split_criteria = channel_info.split_criteria
                self.label_split_channel = i

        # Check if the label can still be split
        if self.is_splittable(split_params.split_threshold):
            # Create a list of keys from splittable_labels
            keys = [l.label_split_criteria for l in splittable_labels]

            # Determine the insertion index using the float value of self.label_split_criteria
            index = bisect_right(keys, self.label_split_criteria)

            # Insert self into splittable_labels at the correct position
            splittable_labels.insert(index, self)
        else:
            self.mask = None  # Equivalent to mask.release() in C++

    def split(self, channels, labels, image_size, next_label, splittable_labels, split_params):
        channel = channels[self.label_split_channel]

        # Get threshold value
        threshold_value = get_channel_threshold(channel, self.channel_infos[self.label_split_channel], self.mask, split_params)

        # Prepare masks
        raw_flood_areas = np.zeros(image_size, dtype=np.uint8)
        flood_seeds = []

        # --LOW INTENSITY REGION--
        low_mask = cv2.threshold(channel, threshold_value, 1, cv2.THRESH_BINARY_INV)[1]
        low_mask = cv2.bitwise_and(low_mask, self.mask.astype(np.uint8))  

        # Connected components analysis
        CCL_Type = cv2.CCL_DEFAULT
        num_labels, cc_labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(low_mask, 4, cv2.CV_32S)
        has_low_labels = False
        flood_seeds = []
        low_mask = None  # Release mask

        for cc_label in range(1, num_labels):
            area = cc_stats[cc_label, cv2.CC_STAT_AREA]
            if area < self.child_min_size:
                raw_flood_areas[cc_labels == cc_label] = 3
                continue
            
            left = cc_stats[cc_label, cv2.CC_STAT_LEFT]
            top = cc_stats[cc_label, cv2.CC_STAT_TOP]
            width = cc_stats[cc_label, cv2.CC_STAT_WIDTH]
            height = cc_stats[cc_label, cv2.CC_STAT_HEIGHT]

            flood_seed = None
            # Find the first valid pixel in the component
            for dx in range(width):
                for dy in range(height):
                    if cc_labels[top + dy, left + dx] == cc_label:
                        flood_seed = (left + dx, top + dy)
                        break
                if flood_seed:
                    break

            raw_flood_areas[cc_labels == cc_label] = 2
            flood_seeds.append(flood_seed)
            has_low_labels = True

        if not has_low_labels:
            return self.interrupt_split(labels, splittable_labels, split_params)

        # --HIGH INTENSITY REGION--
        high_mask = cv2.threshold(channel, threshold_value, 1, cv2.THRESH_BINARY)[1]
        high_mask = cv2.bitwise_and(high_mask, self.mask.astype(np.uint8))

        num_labels, cc_labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(high_mask, 4, cv2.CV_32S)
        has_high_labels = False
        flood_seeds = []
        high_mask = None  # Release mask

        for cc_label in range(1, num_labels):
            area = cc_stats[cc_label, cv2.CC_STAT_AREA]
            if area < self.child_min_size:
                raw_flood_areas[cc_labels == cc_label] = 3
                continue
            
            left = cc_stats[cc_label, cv2.CC_STAT_LEFT]
            top = cc_stats[cc_label, cv2.CC_STAT_TOP]
            width = cc_stats[cc_label, cv2.CC_STAT_WIDTH]
            height = cc_stats[cc_label, cv2.CC_STAT_HEIGHT]

            flood_seed = None
            # Find the first valid pixel in the component
            for dx in range(width):
                for dy in range(height):
                    if cc_labels[top + dy, left + dx] == cc_label:
                        flood_seed = (left + dx, top + dy)
                        break
                if flood_seed:
                    break

            raw_flood_areas[cc_labels == cc_label] = 4
            flood_seeds.append(flood_seed)
            has_high_labels = True

        if not has_high_labels:
            return self.interrupt_split(labels, splittable_labels, split_params)
        
        self.mask = None  # Release mask

        # --Flood Filling--
        flooded_first = False
        for seed in flood_seeds:
            y, x = seed  # Unpack the seed coordinates

            # Ensure y and x are within valid range
            if not (0 <= y < raw_flood_areas.shape[0] and 0 <= x < raw_flood_areas.shape[1]):
                # print(f"Skipping out-of-bounds seed: {seed}")
                continue  # Skip invalid coordinates

            if raw_flood_areas[y, x] == 0:  
                continue  # Already merged
    
            flood_size, _, _, _ = cv2.floodFill(raw_flood_areas, None, seed, 6, 1, 1, cv2.FLOODFILL_FIXED_RANGE)
            flood_mask = (raw_flood_areas == 6)
            raw_flood_areas[flood_mask] = 0

            # Create child label
            child_label_id = next_label if flooded_first else self.id
            child_label = Label(channels, flood_mask, flood_size, child_label_id, split_params)

            if flooded_first:
                labels[flood_mask] = child_label_id

            if child_label.is_splittable(split_params.split_threshold):
                splittable_labels.append(child_label)
                splittable_labels.sort(key=lambda lbl: lbl.label_split_criteria, reverse=True)  # Maintain sorted order

            flooded_first = True


def histogram_bin_to_threshold(bin_value, channel_bins, min_val, max_val):
    threshold = min_val + 0.5 * (((max_val - min_val + 1) * (2 * bin_value + 1) / channel_bins) - 1)
    return int(threshold)

def get_channel_threshold(channel, channel_info, mask, split_params):
    channel_bins = min(split_params.histogram_bins, channel_info.width)
    hist_range = [float(channel_info.min), float(channel_info.max) + 1]

    hist = cv2.calcHist([channel], [0], mask.astype(np.uint8), [channel_bins], hist_range)
    hist = hist.T

    # Apply Laplacian filter to histogram to enhance edges
    kernel = np.array([[1, -2, 1]], dtype=np.float32)  # 1D Laplacian
    response_hist = cv2.filter2D(hist, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    # Compute weights for balanced partitions
    balanced_weights = np.zeros(hist.shape, dtype=np.float32)
    balanced_weights[0, 0] = hist[0, 0]

    for b in range(1, balanced_weights.shape[1]):
        balanced_weights[0, b] = balanced_weights[0, b - 1] + hist[0, b]

    max_weight = balanced_weights[0, -1]
    mean_weight = max_weight / 2

    for b in range(balanced_weights.shape[1]):
        value = balanced_weights[0, b]
        weight = 1 / (pow((mean_weight - value) / mean_weight * 2.0, 4) + 1)
        balanced_weights[0, b] = weight

    # Apply weights to response histogram
    response_hist *= balanced_weights

    # Find threshold bin
    threshold_bin = np.argmax(response_hist)
    thresdholdValue = histogram_bin_to_threshold(threshold_bin, channel_bins, channel_info.min, channel_info.max)
    return thresdholdValue

def hhts(image, labels, superpixels, split_threshold, histogram_bins, min_segment_size, color_channels, apply_blur, pre_labels=None):
    size = image.shape[:2]
    channels = get_channels(image, color_channels, apply_blur)
    
    split_params_obj = split_params(
        superpixels=[superpixels],
        split_threshold=split_threshold,
        histogram_bins=histogram_bins,
        min_segment_size=min_segment_size
    )
    # labels = np.zeros(size, dtype=np.int32)
    splittable_labels = []
    next_label = 1
    # print(split_params_obj.superpixels)

    if pre_labels is None:
        pre_labels = np.ones(size, dtype=np.int32)

    while True:
        max_val = np.max(pre_labels)
        if max_val == 0:
            break  # Processed all pre-labels

        pre_label_pre_id = int(max_val)
        pre_label_id = next_label
        next_label += 1

        pre_label_mask = (pre_labels == pre_label_pre_id).astype(np.uint8)
        pre_labels[pre_label_mask == 1] = 0
        pre_label_size = np.count_nonzero(pre_label_mask)

        pre_label = Label(channels, pre_label_mask, pre_label_size, pre_label_id, split_params_obj)
        labels[pre_label_mask == 1] = pre_label_id

        if pre_label.is_splittable(split_params_obj.split_threshold):
            splittable_labels.append(pre_label)

    output_labels = []
    label_counts = []

    # Sorting function (lower_bound equivalent in Python)
    splittable_labels.sort(key=lambda lbl: lbl.label_split_criteria, reverse=True)
    # Main splitting loop
    while (len(split_params_obj.superpixels) > 0 or split_params_obj.superpixels[0] < 0) and len(splittable_labels) > 0:
        worst_label = splittable_labels.pop(0)
        worst_label.split(channels, labels, size, next_label, splittable_labels, split_params_obj)

        # Check for label output
        while len(split_params_obj.superpixels) > 0 and split_params_obj.superpixels[0] >= 0 and next_label > split_params_obj.superpixels[0]:
            split_params_obj.superpixels.pop(0)

            output_labels.append(labels.copy())  # Store segmentation results
            label_counts.append(next_label)

    # Add remaining segmentation levels
    while len(split_params_obj.superpixels) > 0:
        split_params_obj.superpixels.pop(0)

        output_labels.append(labels.copy())
        label_counts.append(next_label)

    return label_counts, output_labels