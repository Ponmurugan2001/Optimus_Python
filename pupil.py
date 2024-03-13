from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def detect_pupil(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use HoughCircles to detect circles in the image (pupil)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50  # Adjust the maxRadius based on your requirement
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Find the circle closest to the center of the image
        center_x, center_y = gray.shape[1] // 2, gray.shape[0] // 2
        min_distance = float('inf')
        selected_circle = None

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            # Calculate the distance from the circle center to the image center
            distance = np.sqrt((center[0] - center_x)**2 + (center[1] - center_y)**2)
            
            # Update the selected circle if it is closer to the center
            if distance < min_distance:
                min_distance = distance
                selected_circle = i
        
        # Draw a red circle around the selected pupil
        if selected_circle is not None:
            center = (selected_circle[0], selected_circle[1])
            radius = selected_circle[2]
            
            # Calculate the diameter
            diameter = 2 * radius
            print("Diameter of the detected Pupil in {}: {:.2f} pixels".format(image_path, diameter))
            
            # Draw the circle on the image
            cv2.circle(img, center, radius, (0, 0, 255), 2)
            
            return diameter, img
        else:
            print("No pupil detected in {}".format(image_path))
    else:
        print("No circles detected in {}".format(image_path))
    return None, None

def process_folder(folder_path):
    # List all files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Set up live plotting
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title('Diameters of Detected Pupils')
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('Diameter (pixels)')
    line, = ax1.plot([], marker='o', color='b')
    
    # Initialize variables for the minimum and maximum diameters
    min_diameter = float('inf')
    max_diameter = 0
    diameters = []

    # Iterate through each image in the folder
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        diameter, img = detect_pupil(image_path)

        # Update the minimum and maximum diameters
        if diameter is not None:
            min_diameter = min(min_diameter, diameter)
            max_diameter = max(max_diameter, diameter)

        # Append the diameter to the list for live plotting
        ratio = min_diameter/max_diameter
        diameters.append(diameter)

        # Update the live plot
        line.set_xdata(range(len(diameters)))
        line.set_ydata(diameters)
        ax1.relim()
        ax1.autoscale_view()
        plt.draw()
        plt.pause(0.1)

        # Display the last processed eye image in the second subplot
        ax2.clear()
        ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax2.axis('off')

        # Display the results below the image
        ax2.text(0.5, -0.28, "Smallest diameter: {:.2f} pixels\nLargest diameter: {:.2f} pixels\nReflex index: {:.2f}".format(min_diameter, max_diameter, ratio),
                 size=10, ha="center", transform=ax2.transAxes)
        
        plt.draw()
        plt.pause(0.1)

    # Turn off interactive mode after processing
    plt.ioff()

    # Display the final live plot
    plt.show()

    # Return the results for further use if needed
    return min_diameter, max_diameter, ratio

def detect_pupil_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return None

    # Set up live plotting
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title('Detected Pupils in Video')
    ax.axis('off')

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply GaussianBlur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use HoughCircles to detect circles in the frame (pupil)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]

                # Draw a red circle around the detected pupil
                cv2.circle(frame, center, radius, (0, 0, 255), 2)

        # Display the frame with detected circles
        ax.clear()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.draw()
        plt.pause(0.1)

    # Turn off interactive mode after processing
    plt.ioff()
    plt.show()

    # Release the video capture object
    cap.release()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return None

    # Initialize variables for the minimum and maximum diameters
    min_diameter = float('inf')
    max_diameter = 0
    diameters = []

    # Set up live plotting
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title('Diameters of Detected Pupils')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Diameter (pixels)')
    line, = ax.plot([], marker='o', color='b')

    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]

                # Calculate the diameter
                diameter = 2 * radius
                print("Diameter of the detected Pupil in frame {}: {:.2f} pixels".format(frame_index, diameter))

                # Update the minimum and maximum diameters
                min_diameter = min(min_diameter, diameter)
                max_diameter = max(max_diameter, diameter)

                # Append the diameter to the list for live plotting
                ratio = min_diameter / max_diameter
                diameters.append(diameter)

                # Update the live plot
                line.set_xdata(range(len(diameters)))
                line.set_ydata(diameters)
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.1)

        frame_index += 1

    # Turn off interactive mode after processing
    plt.ioff()

    # Display the final live plot
    plt.show()

    # Release the video capture object
    cap.release()

    # Return the results for further use if needed
    return min_diameter, max_diameter, ratio

@app.route('/detect-pupil', methods=['POST'])
def detect_pupil_api():
    data = request.json
    folder_path = data.get('folder_path')

    if folder_path:
        min_diameter, max_diameter, ratio = process_folder(folder_path)
        
        # Convert numpy.int32 values to regular Python integers
        min_diameter = min_diameter.item() if min_diameter is not None else None
        max_diameter = max_diameter.item() if max_diameter is not None else None
        ratio = ratio.item() if ratio is not None else None
        
        return jsonify({
            'min_diameter': min_diameter,
            'max_diameter': max_diameter,
            'ratio': ratio
        }), 200
    else:
        return jsonify({'error': 'Folder path not provided'}), 400

@app.route('/detect-pupil-video', methods=['POST'])
def detect_pupil_video_api():
    data = request.json
    video_path = data.get('video_path')

    if video_path:
        min_diameter, max_diameter, ratio = detect_pupil_video(video_path)

        # Convert numpy.int32 values to regular Python integers
        min_diameter = min_diameter.item() if min_diameter is not None else None
        max_diameter = max_diameter.item() if max_diameter is not None else None
        ratio = ratio.item() if ratio is not None else None

        return jsonify({
            'min_diameter': min_diameter,
            'max_diameter': max_diameter,
            'ratio': ratio
        }), 200
    else:
        return jsonify({'error': 'Video path not provided'}), 400

if __name__ == "__main__":
    app.run(debug=True)
