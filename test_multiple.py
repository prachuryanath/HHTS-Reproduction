import os
import cv2
import numpy as np
import time
from hhts import hhts, ColorChannel
from labelutil import get_colored_labels, get_colored_labels_with_image

def test_multiple_images(input_folder, output_folder, num_images=200):
    """
    Process the first `num_images` images in `input_folder` using HHTS segmentation
    and save the results to `output_folder`. Also computes total and average execution time.

    :param input_folder: Path to the folder containing images.
    :param output_folder: Path to the folder where results will be saved.
    :param num_images: Number of images to process (default: 50).
    """
    sp_count = 500
    min_detail_size = 32

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])[:num_images]

    total_time = 0  # Track total execution time
    num_processed = 0  # Track number of successfully processed images

    for idx, image_name in enumerate(image_files):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Error: Could not load {image_name}. Skipping...")
            continue

        # Initialize labels matrix
        labels = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)

        # Measure execution time for this image
        start_time = time.time()

        # Perform HHTS segmentation
        label_count = hhts(
            image,
            labels,
            sp_count,
            split_threshold=0,
            histogram_bins=64,
            min_segment_size=min_detail_size,
            color_channels=ColorChannel.RGB | ColorChannel.LAB | ColorChannel.HSV,
            apply_blur=False,
            pre_labels=None,
        )

        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        num_processed += 1

        # print(f"[{idx+1}/{len(image_files)}] Processed {image_name} in {elapsed_time:.4f} seconds")

        # Generate colored label images
        mean_labels = get_colored_labels_with_image(labels, image)
        random_labels = get_colored_labels(labels)

        # Save results
        cv2.imwrite(os.path.join(output_folder, f"{image_name}_MeanLabels.jpg"), mean_labels)
        cv2.imwrite(os.path.join(output_folder, f"{image_name}_RandomLabels.jpg"), random_labels)

    # Compute total and average time
    if num_processed > 0:
        average_time = total_time / num_processed
        print(f"\nTotal processing time: {total_time:.4f} seconds")
        print(f"Average time per image: {average_time:.4f} seconds")
    else:
        print("\nNo images were processed.")

    print(f"Processing complete! Results saved in '{output_folder}'.")

# Example usage:
input_folder = "BSDS300/images/train"
output_folder = "BSDS300/results/train"
test_multiple_images(input_folder, output_folder, num_images=200)