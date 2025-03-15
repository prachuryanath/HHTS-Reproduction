import cv2
import numpy as np
import time
from hhts import hhts, ColorChannel
from labelutil import get_colored_labels, get_colored_labels_with_image

def test_single_level():
    image_path = "BSDS300/images/train/8049.jpg"  # Replace with your image path
    sp_count = 500
    min_detail_size = 64

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Could not load image.")
        return

    # Initialize labels matrix
    labels = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)

    # Measure execution time
    start_time = time.time()

    # Perform HHTS segmentation
    label_count = hhts(
        image,
        labels,
        sp_count,
        split_threshold= 0,
        histogram_bins=32,
        min_segment_size=min_detail_size,
        color_channels=ColorChannel.RGB | ColorChannel.LAB | ColorChannel.HSV,
        apply_blur=False,
        pre_labels=None,
    )

    elapsed_time = time.time() - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")

    # Display results
    mean_labels = get_colored_labels_with_image(labels, image)
    random_labels = get_colored_labels(labels)

    cv2.imshow("Mean Labels", mean_labels)
    cv2.imshow("Random Labels", random_labels)
    cv2.imshow('img',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_single_level()
    # test_multi_level()
    # test_autotermination()