# database.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import hashlib
from datetime import datetime

class DatabaseManager:
    def __init__(self):
        self.client = QdrantClient("localhost", port=6333)
        self.init_collections()
    
    def init_collections(self):
        """Initialize Qdrant collections"""
        try:
            self.client.create_collection(
                collection_name="users",
                vectors_config=VectorParams(size=128, distance=Distance.COSINE)
            )
        except Exception as e:
            print(f"Collection might already exist: {e}")
    
    def add_user(self, email, password, face_embeddings):
        """Add new user to database"""
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        user_point = PointStruct(
            id=hash(email),
            vector=face_embeddings,
            payload={
                "email": email,
                "password": hashed_password,
                "created_at": str(datetime.now())
            }
        )
        
        self.client.upsert(
            collection_name="users",
            points=[user_point]
        )
    
    def check_user_exists(self, email):
        """Check if user already exists"""
        search_result = self.client.scroll(
            collection_name="users",
            scroll_filter={"must": [{"key": "email", "match": {"value": email}}]}
        )
        return len(search_result[0]) > 0
    
    def verify_user(self, email, password):
        """Verify user credentials"""
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        search_result = self.client.scroll(
            collection_name="users",
            scroll_filter={
                "must": [
                    {"key": "email", "match": {"value": email}},
                    {"key": "password", "match": {"value": hashed_password}}
                ]
            }
        )
        return len(search_result[0]) > 0