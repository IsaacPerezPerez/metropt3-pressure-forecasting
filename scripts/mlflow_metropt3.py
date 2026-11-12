"""MLflow pyfunc for the frozen MetroPT-3 model."""
import json
import mlflow.pyfunc
import torch
from inference_metropt3 import predict_readings
from train_metropt3 import Model


class MetroModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        checkpoint = torch.load(context.artifacts['checkpoint'], map_location='cpu', weights_only=True)
        self.scaler = json.loads(open(context.artifacts['scaler'], encoding='utf-8').read())
        self.model = Model('LSTM', 6).eval()
        self.model.load_state_dict(checkpoint['model_state'])
        torch.set_num_threads(4)

    def predict(self, context, model_input, params=None):
        return [predict_readings(self.model, self.scaler, readings) for readings in model_input['readings']]
