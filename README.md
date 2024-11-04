# TEXT-TO-SQL
TEXT-TO-SQL : Chatbot to translate natural language into SQL queries with Transformers (DEEP LEARNING APPROACH)

This project demonstrates two approaches to building a chatbot capable of converting natural language questions into SQL queries: a custom-built model from scratch and a fine-tuned transfer learning model. 

## Project Overview

- **Version 1**: A text-to-SQL model built entirely from scratch, which includes:
  - Creating and processing a custom dataset with natural language questions and SQL queries.
  - Implementing the training, validation, and testing processes from the ground up.
  - Model development without leveraging pre-trained transformers.

- **Version 2**: A transfer learning approach using the T5-small transformer model:
  - Fine-tuning the T5 model specifically for the text-to-SQL task.
  - Deploying the fine-tuned model in a web application using Streamlit for real-time interaction.

## Project Structure

### 1. Data Preparation

Both versions use a dataset of natural language questions and corresponding SQL queries:

- **Dataset**: Consists of two columns: `question` (the natural language query) and `answer` (the SQL statement).
- **Processing**:
  - **Version 1**: Data is preprocessed and tokenized manually.
  - **Version 2**: The T5 tokenizer handles tokenization for efficient transformer input.

### 2. Model Training

- **Version 1 (From Scratch)**:
  - Model architecture is built to handle sequence-to-sequence translation of natural language to SQL.
  - Training, validation, and testing loops are custom-coded to fine-tune the model.
  
- **Version 2 (Transfer Learning)**:
  - Utilizes the pre-trained **T5-small transformer** for sequence generation tasks.
  - Fine-tuning is performed by adjusting hyperparameters (e.g., learning rate, batch size) for optimal text-to-SQL performance.
  - Model achieved a training loss of approximately **0.0105** and a validation loss of **0.0217**.

### 3. Evaluation and Testing

- **Version 1**: Model performance is evaluated on a separate test set to measure accuracy and generalization.
- **Version 2**: The fine-tuned T5 model is tested on various real-world questions to assess its SQL generation capabilities. This version is then deployed for interactive use on Streamlit.

## Deployment

The fine-tuned T5 model is deployed in a Streamlit web app, providing users with a user-friendly interface to convert natural language questions into SQL in real time. This deployment demonstrates the model’s practical application for SQL generation tasks.
![image](https://github.com/houda-moudni/TEXT-TO-SQL/blob/main/static/SQL_Y_interface.png)

### Try the Web App
Access the live Streamlit app here: [Text-to-SQL Chatbot](https://text-to-sql-bot.streamlit.app/)


This project highlights the power of both custom-built models and transfer learning in NLP. By fine-tuning a T5 transformer, we successfully created a chatbot capable of transforming natural language questions into SQL queries, making it a valuable tool for non-technical users and database querying automation.


