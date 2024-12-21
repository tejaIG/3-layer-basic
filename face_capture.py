# face_capture.py
import cv2
import os
import numpy as np
import face_recognition

class FaceCapture:
    def __init__(self):
        self.face_directions = [
            "front", "left", "right", "up", "down",
            "slight left", "slight right", "slight up", "slight down"
        ]
        
    def capture_face_dataset(self, user_email):
        """Capture face dataset with different poses"""
        camera = cv2.VideoCapture(0)
        dataset = []
        
        for direction in self.face_directions:
            images = self._capture_direction(camera, direction)
            if images:
                dataset.extend(images)
                
        camera.release()
        return self._process_dataset(dataset, user_email)
    
    def _capture_direction(self, camera, direction):
        """Capture images for a specific direction"""
        images = []
        count = 0
        max_images = 2  # 2 images per direction
        
        while count < max_images:
            ret, frame = camera.read()
            if not ret:
                break
                
            # Display instruction
            cv2.putText(frame, f"Look {direction}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Detect face
            face_locations = face_recognition.face_locations(frame)
            if face_locations:
                images.append(frame)
                count += 1
                
            cv2.imshow("Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        return images
    
    def _process_dataset(self, images, user_email):
        """Process captured images and save them"""
        if not os.path.exists(f"user_data/{user_email}"):
            os.makedirs(f"user_data/{user_email}")
            
        face_embeddings = []
        for i, image in enumerate(images):
            # Get face encoding
            encoding = face_recognition.face_encodings(image)
            if encoding:
                face_embeddings.append(encoding[0])
                # Save image
                cv2.imwrite(f"user_data/{user_email}/face_{i}.jpg", image)
                
        # Return average face embedding
        return np.mean(face_embeddings, axis=0) if face_embeddings else None