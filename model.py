from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Directory where the model is saved
save_directory = 'Transfer_Learning/t5_text2sql'
model_name = 't5-small'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(save_directory).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def generate_sql(query):
    input_text = "translate English to SQL: " + query
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
