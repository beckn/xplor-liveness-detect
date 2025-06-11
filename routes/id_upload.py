import pytesseract
from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2
import face_recognition
from io import BytesIO
from fastapi.responses import JSONResponse
import re

router = APIRouter()

def read_image(file: UploadFile) -> np.ndarray:
    """Reads an image file and returns it as a NumPy array."""
    image_stream = BytesIO(file.file.read())
    image = np.frombuffer(image_stream.getvalue(), dtype=np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

def extract_text_from_image(image: np.ndarray) -> str:
    """Extracts text from an image using Tesseract OCR."""
    # Convert image to grayscale for better text detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_image)
    return text.strip()

def is_valid_document(text: str) -> bool:
    cleaned_text = text.strip()  
    return len(re.findall(r'\w', cleaned_text)) >= 10


def get_face(image: np.ndarray) -> np.ndarray:
    """Detects the largest face in an image and returns it as a cropped NumPy array."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None  # No face detected

    # Select the largest detected face
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    return image[y:y+h, x:x+w]  # Return cropped face region

def extract_face_encoding(image: np.ndarray) -> np.ndarray:
    """Extracts face encoding if a face is found, else returns None."""
    face = get_face(image)
    if face is None:
        return None  # No face found
    
    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_face)
    return encodings[0] if encodings else None  # Return encoding if found

def calculate_match_percentage(face_distance: float) -> float:
    """Converts face distance to match percentage and applies a 10% confidence boost."""
    match_percentage = max(0, (1 - face_distance) * 100)  # Convert distance to percentage
    
    # Apply 10% confidence boost, ensuring the match does not exceed 100%
    boosted_percentage = min(match_percentage * 1.10, 100)  

    return round(boosted_percentage, 2)

@router.post("/upload")
async def compare_faces(id_image: UploadFile = File(...), user_image: UploadFile = File(...)):
    """Compares faces and returns a match percentage."""
    id_image = read_image(id_image)
    user_image = read_image(user_image)

    document_text = extract_text_from_image(id_image)
    if not is_valid_document(document_text):
        return JSONResponse(
            status_code=400,
            content={"message": "Upload a valid document to complete the verification process."}
        )

    id_encoding = extract_face_encoding(id_image)
    user_encoding = extract_face_encoding(user_image)

    if id_encoding is None or user_encoding is None:
        return JSONResponse(
            status_code=400,
            content={"message": "Face does not match the uploaded document."}
        )
    face_distance = face_recognition.face_distance([id_encoding], user_encoding)[0]
    match_percentage = calculate_match_percentage(face_distance)

    return {
        "match_percentage": match_percentage,
        "face_distance": round(face_distance, 4)
    }
