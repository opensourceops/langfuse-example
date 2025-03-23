"""
A Gradio-based chat interface for Groq LLMs with Langfuse observability.

This application demonstrates:
1. Integration with Groq's LLMs using OpenAI-compatible API
2. Chat interface using Gradio
3. Observability and tracing using Langfuse
4. Model switching capabilities
"""

import os
import gradio as gr
import uuid
from typing import List, Tuple, Optional, Dict, Any
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from langfuse.openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = {
    "LANGFUSE_PUBLIC_KEY": "Get from https://cloud.langfuse.com",
    "LANGFUSE_SECRET_KEY": "Get from https://cloud.langfuse.com",
    "LANGFUSE_HOST": "Default: https://cloud.langfuse.com",
    "GROQ_API_KEY": "Get from https://console.groq.com"
}

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables:\n" + 
        "\n".join([f"- {var}: {required_env_vars[var]}" for var in missing_vars])
    )

# Configuration
CONFIG = {
    "temperature": 0.7,
    "max_tokens": 1024,
    "system_prompt": "You are a helpful AI assistant powered by Groq's LLMs."
}

# Initialize Langfuse
langfuse = Langfuse()

# Initialize OpenAI client with Groq endpoint
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

# Define available models
GROQ_MODELS = {
    "LLaMA 3.3 70B Versatile": "llama3-70b-8192",
    "LLaMA 3.1 8B Instant": "llama3-8b-8192",
    "Mixtral 24B": "mistral-saba-24b",
    "Gemma 2 9B": "gemma2-9b-it"
}

# Global variables for session and trace management
session_id: Optional[str] = None
current_trace_id: Optional[str] = None
current_model: str = "llama3-70b-8192"  # default model

def set_new_session_id() -> None:
    """Create a new session ID for tracking conversation threads."""
    global session_id
    session_id = str(uuid.uuid4())

# Initialize session
set_new_session_id()

def update_model(model_name: str) -> str:
    """
    Update the current model being used.
    
    Args:
        model_name: The display name of the model to use
        
    Returns:
        str: Confirmation message of the model update
    """
    global current_model
    current_model = GROQ_MODELS[model_name]
    return f"Model updated to: {model_name}"

@observe(as_type="generation")
async def create_response(
    prompt: str,
    history: List[Tuple[str, str]],
    model_name: str
) -> List[Tuple[str, str]]:
    """
    Generate response using Groq's LLM via OpenAI SDK with Langfuse tracing.
    
    Args:
        prompt: The user's input message
        history: Chat history as a list of (user, assistant) message tuples
        model_name: The display name of the model to use
        
    Returns:
        List[Tuple[str, str]]: Updated chat history with the new response
    """
    global current_trace_id
    current_trace_id = langfuse_context.get_current_trace_id()

    # Add session_id to Langfuse Trace
    global session_id
    langfuse_context.update_current_trace(
        name="groq_chat",
        session_id=session_id,
        input=prompt,
        metadata={
            "model": GROQ_MODELS[model_name],
            "temperature": CONFIG["temperature"],
            "max_tokens": CONFIG["max_tokens"]
        }
    )

    # Convert history from Gradio format to OpenAI format
    messages = [{"role": "system", "content": CONFIG["system_prompt"]}]
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": prompt})

    try:
        # Get completion from Groq
        completion = client.chat.completions.create(
            model=GROQ_MODELS[model_name],
            messages=messages,
            temperature=CONFIG["temperature"],
            max_tokens=CONFIG["max_tokens"]
        )
        assistant_message = completion.choices[0].message.content or ""

        # Update Langfuse trace with output
        langfuse_context.update_current_trace(
            output=assistant_message,
        )

        # Return history in Gradio format (list of tuples)
        history.append((prompt, assistant_message))
        return history

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history.append((prompt, error_msg))
        return history

async def respond(
    prompt: str,
    history: Optional[List[Tuple[str, str]]],
    model_name: str
) -> List[Tuple[str, str]]:
    """Handle chat interactions."""
    history = history or []
    return await create_response(prompt, history, model_name)

def handle_like(data: gr.LikeData) -> None:
    """
    Handle user feedback through likes/dislikes.
    
    Args:
        data: Gradio like data containing the feedback
    """
    global current_trace_id
    if current_trace_id:
        if data.liked:
            langfuse.score(value=1, name="user-feedback", trace_id=current_trace_id)
        else:
            langfuse.score(value=0, name="user-feedback", trace_id=current_trace_id)

async def handle_retry(
    history: List[Tuple[str, str]],
    retry_data: gr.RetryData,
    model_name: str
) -> List[Tuple[str, str]]:
    """
    Handle retry requests for regenerating responses.
    
    Args:
        history: Current chat history
        retry_data: Gradio retry data
        model_name: The display name of the model to use
        
    Returns:
        List[Tuple[str, str]]: Updated chat history with the regenerated response
    """
    if not history:
        return []
    
    # Remove the last exchange
    new_history = history[:-1]
    # Get the last user message
    last_user_message = history[-1][0] if history else ""
    
    if last_user_message:
        return await respond(last_user_message, new_history, model_name)
    return new_history

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Groq Chat with Langfuse Tracing")
    gr.Markdown("""
    This demo showcases:
    - Multiple Groq LLM models
    - Chat interface with history
    - Langfuse observability
    - User feedback collection
    """)
    
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=list(GROQ_MODELS.keys()),
            value="LLaMA 3.3 70B Versatile",
            label="Select Model",
            info="Choose the Groq model to use"
        )
        model_status = gr.Markdown("Current Model: LLaMA 3.3 70B Versatile")
    
    chatbot = gr.Chatbot(
        label="Chat",
        show_copy_button=True,
        avatar_images=(None, "https://groq.com/favicon.ico"),
        height=500
    )
    
    with gr.Row():
        prompt = gr.Textbox(
            placeholder="Type your message here...",
            label="Input",
            scale=9
        )
        submit = gr.Button("Send", scale=1)
    
    with gr.Row():
        clear = gr.Button("Clear Chat")
        retry = gr.Button("Retry Last")

    # Set up event handlers
    model_dropdown.change(
        fn=lambda x: f"Current Model: {x}",
        inputs=[model_dropdown],
        outputs=[model_status]
    )
    
    prompt.submit(respond, [prompt, chatbot, model_dropdown], [chatbot])
    submit.click(respond, [prompt, chatbot, model_dropdown], [chatbot])
    clear.click(lambda: [], None, chatbot)
    clear.click(set_new_session_id)
    retry.click(handle_retry, [chatbot, model_dropdown], [chatbot])
    chatbot.like(handle_like, None, None)

if __name__ == "__main__":
    demo.launch(share=True)
