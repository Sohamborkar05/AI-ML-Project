from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import imutils
import easyocr
import pandas as pd

app = Flask(__name__)

# Path to Haar Cascade for license plate detection
harcascade = "model/haarcascade_russian_plate_number.xml"

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Global variable to store detected license plate
detected_plate = None
matched_data = None  # Variable to store matched data

# Function to find a row based on a keyword (license plate text)
def find_row_by_keyword(file_path, sheet_name, column_name, keyword):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    matching_row = df[df[column_name].str.contains(keyword, na=False, case=False)]
    if not matching_row.empty:
        return matching_row  # Return the matching row(s)
    else:
        return None  # Return None if no match is found

# Generate frames from the camera
def generate_frames():
    global detected_plate, matched_data
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # width
    cap.set(4, 480)  # height
    plate_found = False  # Flag to break the loop when a plate is found

    while True: 
        success, img = cap.read()
        if not success:
            break

        plate_cascade = cv2.CascadeClassifier(harcascade)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        for (x, y, w, h) in plates:
            area = w * h
            if area > 500:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img_roi = img[y:y + h, x:x + w]
                gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

                # Process the detected plate image
                result = reader.readtext(gray)
                if result:
                    detected_plate = result[0][-2]
                    print(f"Detected License Plate: {detected_plate}")

                    # Search the Excel file for the detected license plate
                    file_path = 'numberr.xlsx'  # Replace with the actual file path
                    sheet_name = 'Sheet1'       # Replace with your actual sheet name
                    column_name = 'number'      # Replace with the column name where license plates are stored

                    match_found = find_row_by_keyword(file_path, sheet_name, column_name, detected_plate)
                    
                    if match_found is not None:
                        matched_data = match_found.to_dict(orient='records')  # Store matched data
                        plate_found = True  # Set the flag to stop the detection loop
                        break  # Exit the loop since we found the plate
                    else:
                        matched_data = None
                        print("No matching data found")
        
        if plate_found:
            # Stop the camera feed and detection after plate is found
            break

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        # Yield the frame in the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # Release the camera when done

# Route to provide the detected license plate and matched data in JSON format
@app.route('/get_detected_plate')
def get_detected_plate():
    if detected_plate:
        if matched_data:
            return jsonify({'plate': detected_plate, 'data': matched_data})
        else:
            return jsonify({'plate': detected_plate, 'data': []})
    else:
        return jsonify({'plate': '', 'data': []})

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
