import ollama
import streamlit as st

st.set_page_config(page_title='Ollama LLM')
st.title("LLM Chatbot")

if 'chat_hist' not in st.session_state:
    st.session_state.chat_hist = []

prompt = st.chat_input('Enter Your Prompt')

for message in st.session_state.chat_hist:
    with st.chat_message(message['role']):
        st.write(message['content'])

if prompt:
    st.session_state.chat_hist.append({'role':'user','content':prompt})
    with st.chat_message('user'):
        st.write(prompt)
    with st.spinner('Thinking...'):
        result = ollama.chat(model='llama3.2',messages=st.session_state.chat_hist)
        response = result['message']['content']
        st.session_state.chat_hist.append({'role':'assistant','content':response})
    with st.chat_message('assistant'):
        st.write(response)
 
