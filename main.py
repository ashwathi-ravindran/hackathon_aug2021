import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from ml_utils import load_model, predict, retrain
from typing import List

# defining the main app
app = FastAPI(title="Bug classifier", docs_url="/")

# calling the load_model during startup.
# this will train the model and keep it loaded for prediction.
app.add_event_handler("startup", load_model)

# class which is expected in the payload
class QueryIn(BaseModel):
    lOBlank: float
    uniq_Op: float
    loc: float
    d: float
    lOComment: float

# class which is returned in the response
class QueryOut(BaseModel):
    bug_class: str
    timestamp: datetime

# class which is expected in the payload while re-training
class FeedbackIn(BaseModel):
    lOBlank: float
    uniq_Op: float
    loc: float
    d: float
    lOComment: float
    bug_class: str

# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/predict_bug", response_model=QueryOut, status_code=200)
# Route to do the prediction using the ML model defined.
# Payload: QueryIn containing the parameters
# Response: QueryOut containing the flower_class predicted (200)
def predict_bug(query_data: QueryIn):
    output = {"bug_class": predict(query_data),"timestamp":datetime.now()}
    return output

@app.post("/feedback_loop", status_code=200)
# Route to further train the model based on user input in form of feedback loop
# Payload: FeedbackIn containing the parameters and correct flower class
# Response: Dict with detail confirming success (200)
def feedback_loop(data: List[FeedbackIn]):
    retrain(data)
    return {"detail": "Feedback loop successful"}


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True)