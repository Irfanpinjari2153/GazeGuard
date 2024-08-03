import cv2
import pandas as pd
from openpyxl import load_workbook
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import datetime

# Load the pre-trained face recognition model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model.yml')

# Verify the file path and ensure 'dataset.xlsx' exists
file_path = 'dataset.xlsx'  # Update with the correct path if necessary
try:
    # Load the dataset
    dataset = pd.read_excel(file_path, engine='openpyxl')
    print("Dataset loaded successfully.")  # Confirmation message
except FileNotFoundError:
    print(f"File not found: {file_path}. Please check the file path.")
    # Handle the error appropriately, e.g., exit the program or prompt for a valid path
    # For this example, we will create a sample dataset if the file is not found
    dataset = pd.DataFrame(columns=['Name']) # Create an empty dataframe with a 'Name' column
    dataset.to_excel(file_path, index=False, engine='openpyxl') # Save the empty dataframe to an Excel file

# Function to capture an image from the camera
def capture_image():
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    if ret:
        cv2.imshow("Capture", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        camera.release()
        return frame
    else:
        print("Could not access the camera.")
        return None

# Function to perform facial recognition
def recognize_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No face detected.")
        return None

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi_gray)

        # Check if the confidence is below a certain threshold
        if confidence < 50:
            return label
        else:
            print("Unknown face.")
            return None

# Function to mark attendance in Excel sheet
def mark_attendance(label):
    today = pd.to_datetime('today').strftime('%Y-%m-%d')
    if today not in dataset.columns:
        dataset[today] = np.nan

    # Check if label exists in the dataset
    if label in dataset['Name'].values:
        dataset.loc[dataset['Name'] == label, today] = 'Present'
    else:
        print(f"Label {label} not found in dataset.")

    # Save the updated dataset
    dataset.to_excel(file_path, index=False, engine='openpyxl')

    print("Attendance marked for label:", label)

# Function to send the attendance sheet through email
def send_email():
    from_addr = 'your_email@example.com'  # Replace with your actual email address
    to_addr = 'recipient@example.com'  # Replace with the recipient's email address
    password = 'your_email_password'  # Replace with your actual email password or app-specific password
    msg = MIMEMultipart()
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = 'Attendance Sheet'

    body = 'Please find attached the attendance sheet.'
    msg.attach(MIMEText(body, 'plain'))

    # Attach the Excel file
    with open(file_path, 'rb') as f:
        attachment = MIMEApplication(f.read(), _subtype="xlsx")
        attachment.add_header('Content-Disposition', 'attachment', filename='dataset.xlsx')
        msg.attach(attachment)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()

    # Handle potential authentication errors
    try:
        server.login(from_addr, password)
        text = msg.as_string()
        server.sendmail(from_addr, to_addr, text)
        print("Email sent successfully!")  # Confirmation message
    except smtplib.SMTPAuthenticationError as e:
        print("Authentication error:", e)  # Print the error message
    finally:
        server.quit()

# Example usage:
frame = capture_image()
if frame is not None:
    label = recognize_face(frame)
    if label is not None:
        mark_attendance(label)
        send_email()
