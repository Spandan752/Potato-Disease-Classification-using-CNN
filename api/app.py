from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import uvicorn


app = FastAPI()

model = tf.keras.models.load_model("D:\Projects\Potato_Disease\models\CNN_model1.keras")

class_names = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/")
def check_health():
    return {"status": "ok"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    prediction = model.predict(img_batch)

    predicted_class = class_names[np.argmax(prediction[0])]

    confidence = np.max(prediction[0])

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)