import streamlit as st
from model import generate_sql

# Sidebar setup
with st.sidebar:
    st.markdown("### About SQL-y")
    st.write("SQL-y is a chatbot powered by Transformer that can assist you with SQL queries.")
    st.write("Feel free to ask any SQL-related questions!")
    st.markdown("---")  # Add a horizontal line for separation
    st.write("*Made by:*")
    st.write("Houda Moudni")
    st.write("Chadi Mountassir")


image_column, title_column = st.columns([1, 2]) 
with image_column:
    st.image("Transfer_Learning/static/logo.jpg", use_column_width=True) # Adjust the width ratio as needed
with title_column:
    st.title("SQL-y")
    st.caption("A SQLBot powered by Transformer")
  # Adjust use_column_width as needed


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.text_input("You", value=msg["content"], key=msg["content"])
    else:
        st.text_input("SQLy", value=msg["content"], key=msg["content"])

# Chat input field
if prompt := st.text_input("You", placeholder="Type your question here..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    print(prompt)
    st.session_state.messages.append({"role": "assistant", "content": f"{generate_sql(prompt)}"})



