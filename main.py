import cv2
import numpy as np
from tensorflow.keras.models import load_model
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configuration
SENDER_EMAIL = "shrinkhalshrinkhal@gmail.com"
RECEIVER_EMAILS = [
    "privatehvro@gmail.com",
    "shrinkhalshrinkhal22@gmail.com",
    "adityabhatt78910@gmail.com",
    "lakshitatak1@gmail.com",
    "iishasharrma@gmail.com"
]
EMAIL_PASSWORD = "pptsbbotjoqgwdkp" 
model = load_model('my_model_.keras')
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (1280, 720))  # Resize to match model's input size
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
def send_email(frame_count, accident_probability):
    try:
        subject = "Accident Detected!"
        body = f"An accident was detected in frame {frame_count} with a probability of {accident_probability:.2f}."
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = ", ".join(RECEIVER_EMAILS)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())
        print("Email notification sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to predict accidents in a video
def detect_accident(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    email_sent = False  # Flag to track if the email has been sent

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1
        processed_frame = preprocess_frame(frame)
        
        try:
            prediction = model.predict(processed_frame)  # Use the preprocessed frame directly
        except ValueError as e:
            print(f"Error while predicting frame {frame_count}: {e}")
            break

        # Assuming the model outputs a probability for 'accident'
        accident_probability = prediction[0][0]  # Adjust index if your model has other outputs

        if accident_probability > 0.6 and not email_sent:  # Threshold for accident detection
            print(f"Accident detected in frame {frame_count} with probability {accident_probability:.2f}")
            send_email(frame_count, accident_probability)  # Send email notification
            email_sent = True  # Set the flag to True to prevent duplicate emails

        # Display the frame (optional, for visualization)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to run the script
if __name__ == "__main__":
    video_path = "acc1.mp4"  # Path to your input video
    detect_accident(video_path)
