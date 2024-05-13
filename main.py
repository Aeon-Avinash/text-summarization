# Load model directly without pipeline
from transformers import pipeline
import torch
import gradio
from litserve import LitAPI, LitServer

class YTSummarizerLit(LitAPI):
    def setup(self, device):
        """
        Load the model via a huggingface pipeline.
        """
        model_name="sshleifer/distilbart-cnn-12-6"
        self.pipeline = pipeline("summarization", model=model_name, device=0 if device=="gpu" else -1, torch_dtype=torch.bfloat16)

    def decode_request(self, request):
        """
        Preprocess the request data (tokenize)
        """
        return request["text"]

    def predict(self, text):
        """
        Perform the inference
        """
        return self.pipeline(text)

    def encode_response(self, output):
        """
        Process the model output into a response dictionary
        """
        return {"output": output}

# START THE SERVER
if __name__ == "__main__":
    api = YTSummarizerLit()
    server = LitServer(api, accelerator="auto")
    server.run(port=8000)