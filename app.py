from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)
CORS(app)

# Load your trained BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load('bert_model_state_dict.pth', map_location=torch.device('cpu'))
state_dict.pop("bert.embeddings.position_ids", None)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_bias', methods=['POST'])
def detect_bias():
    if request.method == 'POST':
        article = request.form['article']

        # Tokenize the input
        inputs = tokenizer(article, return_tensors="pt", truncation=True, padding=True)
        
        # Move inputs to the correct device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Forward pass through the model
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Get predicted class
        predicted_class = torch.argmax(logits, dim=1).item()

        # Convert the predicted class to a bias label
        bias_labels = ["Left", "Center", "Right"]
        predicted_bias = bias_labels[predicted_class]

        # Get the confidence scores for each class (optional)
        confidence_scores = torch.nn.functional.softmax(logits, dim=1).tolist()[0]

        # Prepare the response
        response = {
            'bias': predicted_bias,
            'confidence': round(max(confidence_scores) * 100, 2)  # Assuming you want the confidence of the predicted class
        }

        print("RAN")
        print(response)
        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
