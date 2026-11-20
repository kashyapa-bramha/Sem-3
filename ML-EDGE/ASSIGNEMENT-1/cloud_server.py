"""Lab 1, Step 4a: Serve the MNIST TFLite model through FastAPI."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class ImageRequest(BaseModel):
    # Step 1 already normalized the image; the client sends it as a flat list.
    data: list[Annotated[float, Field(ge=0, le=1)]] = Field(min_length=784, max_length=784)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Finish model setup before the server accepts its first request.
    model_path = Path(__file__).with_name("model.tflite")
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    app.state.interpreter = interpreter
    app.state.input_index = interpreter.get_input_details()[0]["index"]
    app.state.output_index = interpreter.get_output_details()[0]["index"]
    app.state.inference_lock = Lock()
    yield


app = FastAPI(title="Lab1 Cloud Inference", lifespan=lifespan)


@app.get("/")
def health():
    return {"status": "healthy"}


@app.post("/infer")
def infer(request: ImageRequest):
    image = np.asarray(request.data, dtype=np.float32).reshape(1, 28, 28, 1)
    interpreter = app.state.interpreter
    try:
        # Keep each request's input and output together on the shared interpreter.
        with app.state.inference_lock:
            start = perf_counter()
            interpreter.set_tensor(app.state.input_index, image)
            interpreter.invoke()
            server_ms = (perf_counter() - start) * 1000
            scores = interpreter.get_tensor(app.state.output_index)[0]
    except (RuntimeError, ValueError) as error:
        logging.getLogger(__name__).exception("TFLite inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed.") from error

    prediction = int(np.argmax(scores))
    return {
        "prediction": prediction,
        "confidence": round(float(scores[prediction]), 4),
        "server_latency_ms": round(server_ms, 3),
        # The handout reports float32 bytes here; JSON upload size is a client metric.
        "payload_bytes": image.nbytes,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)