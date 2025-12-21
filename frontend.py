"""
Streamlit UI for interacting with a LangGraph-based chatbot.

This module provides a minimal chat interface using Streamlit that:
- Captures user input via Streamlit's chat components
- Maintains conversation history in Streamlit session state
- Sends user messages to a LangGraph-powered chatbot backend
- Renders assistant responses in a conversational format

The UI is intentionally lightweight and designed for rapid
experimentation and local development rather than production deployment
"""

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot

CONFIG={
    'configurable': {
        'thread_id': 'thread-1'
    }
}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    st.session_state['message_history'].append(
        {'role': 'user', 'content': user_input}
    )
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {
                    'messages': [HumanMessage(content=user_input)]
                },
                config=CONFIG,
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
