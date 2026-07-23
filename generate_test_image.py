"""
Zebra Crossing Synthetic Image Generator for EyeCan Testing
Generates a perfectly formatted image matching all HSV and Hough line criteria.
"""

import cv2
import numpy as np

def generate_zebra_crossing():
    # 1. Create a 640x480 dark gray asphalt canvas
    img = np.full((480, 640, 3), (40, 40, 40), dtype=np.uint8)

    # 2. Draw 6 bright white zebra stripes in the lower region (y > 264)
    # Each stripe is 360px wide, 25px thick, spaced 15px apart
    start_y = 280
    stripe_height = 20
    gap = 15
    stripe_width = 400
    start_x = 120

    for i in range(6):
        y1 = start_y + i * (stripe_height + gap)
        y2 = y1 + stripe_height
        x1 = start_x
        x2 = start_x + stripe_width
        
        # Draw pure white rectangle (BGR: 255, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)

    # 3. Save to disk
    filename = "zebra_test.jpg"
    cv2.imwrite(filename, img)
    print(f"Successfully generated test image: '{filename}'!")

    # 4. Display the image on screen
    cv2.imshow("Generated Zebra Crossing Test", img)
    print("Press any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_zebra_crossing()