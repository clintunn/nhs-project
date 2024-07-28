import cv2  # Imports the OpenCV cv2 dependency
from pyzbar.pyzbar import decode  # Imports pyzbar for decoding QR codes
import os

# Using raw string to handle backslashes
image_path = r'C:\Users\User\Desktop\NHS\QRcode.PNG'
output_path = r'C:\Users\User\Desktop\NHS\QRcode_detected.png'

# Print the image path to ensure it's correct
print(f"Image path: {image_path}")  # Error logging to help debug in case of errors with relations to image path

# Load the image
image = cv2.imread(image_path)  # This is a method from the imported cv2 that reads the image from the path

# Check if the image was loaded successfully
if image is None:  # Error handling if the image cannot be loaded
    print(f"Error: Unable to load image at {image_path}")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Decode the QR codes
    decoded_objects = decode(gray_image)

    # Extract and print the data from each QR code
    for obj in decoded_objects:
        data = obj.data.decode('utf-8')
        print('Type:', obj.type)
        print('Data:', data)
        print()

        # Draw the detected QR codes on the image
        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(points)
            points = hull.reshape(-1, 2)
        
        # Draw the lines around the QR code
        for i in range(len(points)):
            cv2.line(image, tuple(points[i]), tuple(points[(i + 1) % len(points)]), (0, 255, 0), 3)

        # Determine the position to place the text
        x = min(point[0] for point in points)
        y = min(point[1] for point in points) - 10  # Position the text slightly above the QR code

        # Put the text (data) on the image
        cv2.putText(image, data, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the image with detected QR codes
    cv2.imwrite(output_path, image)
    print(f"Output image saved at {output_path}")
