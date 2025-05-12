import streamlit as st
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Chatbot Interface",
    page_icon="💬",
    layout="centered"
)

# App title
st.title("💬 Chatbot")
st.markdown("Ask me anything!")

# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to interact with the API
def query_chatbot(question):
    api_url = "http://localhost:8000/api/ask"
    headers = {"Content-Type": "application/json"}
    data = {"question": question}
    
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()["result"]
    except requests.exceptions.RequestException as e:
        return f"Error: Unable to connect to the chatbot API. {str(e)}"
    except (KeyError, json.JSONDecodeError):
        return "Error: Invalid response from the chatbot API."

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_chatbot(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add sidebar with information
with st.sidebar:
    st.subheader("About")
    st.markdown("""
    This is a simple chatbot interface built with Streamlit.
    
    It connects to a local API endpoint that processes your questions 
    and returns answers.
    """)
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()