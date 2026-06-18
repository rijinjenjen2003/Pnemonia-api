from fastapi import FastAPI, UploadFile, File
from keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

model = load_model("pneumonia_model.h5")

@app.get("/")
def home():
    return {"message": "Pneumonia Detection API Running"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))

        # Convert to RGB
        image = image.convert("RGB")

        # CHANGE THIS SIZE TO MATCH model.input_shape
        image = image.resize((150, 150))

        img_array = np.array(image) / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        confidence = float(prediction[0][0])

        if confidence > 0.5:
            result = "Pneumonia"
        else:
            result = "Normal"

        return {
            "prediction": result,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}