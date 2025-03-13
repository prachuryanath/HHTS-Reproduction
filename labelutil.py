import cv2
import numpy as np

def get_random_color():
    """Generate a random color in BGR format."""
    b = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    r = np.random.randint(0, 256)
    return (b, g, r)

def get_colored_labels(input_labels):
    """Assign random colors to labeled segments, with label 0 remaining black."""
    labels = input_labels.astype(np.int32)
    max_label = np.max(labels)
    
    # Initialize output image with black background
    label_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    
    for label in range(1, int(max_label) + 1):
        mask = (labels == label)
        label_image[mask] = get_random_color()
    
    return label_image

def get_colored_labels_with_image(input_labels, input_image):
    """Assign mean color of corresponding regions in input_image to labeled segments."""
    labels = input_labels.astype(np.int32)
    image = input_image.copy()
    max_label = np.max(labels)
    
    # Initialize output image with black background
    label_image = np.zeros_like(image, dtype=np.uint8)
    
    for label in range(1, max_label + 1):
        mask = (labels == label)
        mean_color = cv2.mean(image, mask=mask.astype(np.uint8))[:3]  # Extract BGR mean values
        label_image[mask] = mean_color
    
    return label_image

# Load an example image
# image = cv2.imread("BSDS300/images/train/8049.jpg")  # Replace with your actual image path
# image = cv2.imread("BSDS300/images/train/23080.jpg")  # Replace with your actual image path
image = cv2.imread("BSDS300/images/train/41004.jpg")  # Replace with your actual image path
# image = cv2.resize(image, (100, 300))  # Resize for visualization (optional)

# Generate random labels (for testing purposes)
labels = np.random.randint(0, 5, size=(image.shape[0], image.shape[1]), dtype=np.int32)

# Get the colorized segmentation image
colored_labels = get_colored_labels_with_image(labels, image)

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Colored Labels", colored_labels)
cv2.waitKey(0)
cv2.destroyAllWindows()
