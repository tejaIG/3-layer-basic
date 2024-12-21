# face_utils.py
import face_recognition
import numpy as np

class FaceRecognition:
    def __init__(self):
        self.threshold = 0.6  # Similarity threshold
        
    def verify_face(self, captured_face, stored_embedding):
        """Verify if captured face matches stored embedding"""
        if captured_face is None or stored_embedding is None:
            return False
            
        face_encoding = face_recognition.face_encodings(captured_face)
        if not face_encoding:
            return False
            
        # Compare face encodings
        distance = face_recognition.face_distance([stored_embedding], face_encoding[0])
        return distance[0] <= self.threshold