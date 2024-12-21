# app.py
import streamlit as st
import cv2
import numpy as np
from database import DatabaseManager
from face_capture import FaceCapture
from face_utils import FaceRecognition
import os

# Initialize components
db = DatabaseManager()
face_capture = FaceCapture()
face_recognition_util = FaceRecognition()

def main():
    st.title("Two-Layer Authentication System")
    
    menu = ["Sign Up", "Login"]
    choice = st.sidebar.selectbox("Select Action", menu)
    
    if choice == "Sign Up":
        handle_signup()
    else:
        handle_login()

def handle_signup():
    st.subheader("Create New Account")
    
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Register"):
        if not email or not password:
            st.error("Please fill in all fields")
            return
            
        if db.check_user_exists(email):
            st.error("User already exists!")
            return
            
        st.info("Starting face capture process...")
        st.info("We'll capture your face from different angles. Follow the instructions on the camera window.")
        
        # Capture face dataset
        face_embedding = face_capture.capture_face_dataset(email)
        
        if face_embedding is not None:
            # Store user data
            db.add_user(email, password, face_embedding)
            st.success("Registration successful!")
        else:
            st.error("Failed to capture face properly. Please try again.")

def handle_login():
    st.subheader("Login")
    
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if not email or not password:
            st.error("Please fill in all fields")
            return
            
        # First layer: verify credentials
        if not db.verify_user(email, password):
            st.error("Invalid email or password")
            return
            
        st.info("Credentials verified. Starting face verification...")
        
        # Second layer: face verification
        camera = cv2.VideoCapture(0)
        
        # Get stored face embedding
        user_data = db.client.scroll(
            collection_name="users",
            scroll_filter={"must": [{"key": "email", "match": {"value": email}}]}
        )
        
        if not user_data[0]:
            st.error("User data not found")
            return
            
        stored_embedding = user_data[0][0].vector
        
        # Capture current face
        ret, frame = camera.read()
        camera.release()
        
        if ret:
            if face_recognition_util.verify_face(frame, stored_embedding):
                st.success("Login successful!")
            else:
                st.error("Face verification failed")
        else:
            st.error("Failed to capture face")

if __name__ == "__main__":
    main()