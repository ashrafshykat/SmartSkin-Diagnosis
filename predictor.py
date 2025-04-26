import numpy as np
import cv2
import requests
import tensorflow as tf

class_names = [
    'Acne and Rosacea',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Atopic Dermatitis',
    'Cellulitis Impetigo and other Bacterial Infections',
    'Eczema',
    'Herpes HPV and other STDs',
    'Light Diseases and Disorders of Pigmentation',
    'Exanthems and Drug Eruptions',
    'Lupus and other Connective Tissue diseases',
    'Melanoma Skin Cancer Nevi and Moles',
    'Poison Ivy Photos and other Contact Dermatitis',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Seborrheic Keratoses and other Benign Tumors',
    'Systemic Disease',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Urticaria Hives',
    'Vascular Tumors',
    'Vasculitis',
    'Warts Molluscum and other Viral Infections'
]

# Load the trained model
cnn_model = tf.keras.models.load_model('model_latest.h5') 

def preprocess_image(image):
    """Resizes and preprocesses the input image for model prediction."""
    img_size = (192, 192)  
    image = cv2.resize(image, img_size)  
    image = image[:, :, ::-1]  
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  
    return image

def predict_from_url(image_url):
    """Fetches an image from a URL and predicts the skin disease."""
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            return "Failed to fetch image."

        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            return "Invalid image format."

        processed_image = preprocess_image(image)
        prediction = cnn_model.predict(processed_image, verbose=0)[0]
        predicted_class_index = np.argmax(prediction)
        return class_names[predicted_class_index]

    except Exception as e:
        return f"Error: {str(e)}"

def predict_from_upload(uploaded_file):
    """Reads an uploaded image and predicts the skin disease."""
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return "Invalid image format."

        processed_image = preprocess_image(image)
        prediction = cnn_model.predict(processed_image, verbose=0)[0]
        predicted_class_index = np.argmax(prediction)
        return class_names[predicted_class_index]

    except Exception as e:
        return f"Error: {str(e)}"

def predict_from_realtime():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return "Error: Unable to access webcam."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_image = preprocess_image(frame)

        prediction = cnn_model.predict(processed_image, verbose=0)[0]
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_names[predicted_class_index]

        cv2.putText(frame, f"Prediction: {predicted_class_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Real-Time Skin Disease Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return "Real-time prediction stopped."
