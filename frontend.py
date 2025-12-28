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

import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph_backend import chatbot, retrieve_all_threads

def generate_thread_id():
    """
    Generates and return thread id
    """
    thread_id=uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history']=[]

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

# ****************************** Session Setup ***************************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])

CONFIG={
    'configurable': {
        'thread_id': st.session_state['thread_id']
    },
    'metadata': {
        'thread_id': st.session_state['thread_id']
    },
    'run_name': 'chat_run'
}

# ****************************************** Sidebar UI **************************************************8
st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button(label='New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id']=thread_id
        messages=load_conversation(thread_id=thread_id)

        temp_messages=[]

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})
        st.session_state['message_history']=temp_messages

# **************************** Main UI ********************************************
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

        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name=getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"]=st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True
                        )
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message=st.write_stream(ai_only_stream)

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state['message_history'].append(
        {
            'role': 'assistant',
            'content': ai_message
        }
    )
