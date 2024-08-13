import os
import autogen
from autogen.oai.client import OpenAIWrapper
from autogen.coding import LocalCommandLineCodeExecutor
from typing import  Any, Dict, List, Literal, Optional, Union, Tuple, Annotated
import random
from huggingface_hub import InferenceClient
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
import time

class HuggingFaceClient:
    def __init__(
        self,
        api_key: Optional[Union[str, bool, None]] = None,
        model: Optional[Union[str, None]] = None,
        base_url: Optional[str] = None,
        inference_mode: Optional[Literal["auto", "local", "remote"]] = "auto",
        config: Optional[OpenAIWrapper] = None,
        **kwargs,
    ):
        self._api_key = api_key
        if not self._api_key:
            self._api_key = os.getenv("HF_TOKEN")

        self._default_model = model

        self._inference_client = InferenceClient(model=self._default_model, token=self._api_key)

    def create(self, params: Dict[str, Any]) -> ChatCompletion:
       
        model = params.get("model", None)
               # Convert list of messages to a single string
        input_text = "\n".join([msg['content'] for msg in params["messages"]])

        res = self._inference_client.text_generation(input_text, max_new_tokens=4096)

        # Create ChatCompletion
        message = ChatCompletionMessage(role="assistant", content=res)
        choices = [Choice(finish_reason="stop", index=0, message=message)]

        response_oai = ChatCompletion(
            id=str(random.randint(0, 1000)),
            choices=choices,
            created=int(time.time() * 1000),
            model=model,
            object="chat.completion",
        )

        return response_oai

   
    def message_retrieval(self, response) -> List:
        return [choice.message.content for choice in response.choices]

    def cost(self, response) -> float:
        return 0.0

    @staticmethod
    def get_usage(response) -> Dict:
        return {
            "model": response.model,
        }

# Configuration
config_list = [
    {
        "model": "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8", 
        "model_client_cls": "HuggingFaceClient",
        "base_url": "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8", 
        "api_key": os.getenv("HF_TOKEN")
    },
]

# Create an AssistantAgent that will handle API calls
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,
        "temperature": 0,
    },
)

# Create a UserProxyAgent that doesn't make API calls
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(work_dir="coding"),
    },
)

# Only register the custom model client for the assistant
assistant.register_model_client(model_client_cls=HuggingFaceClient, model="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8", token=os.getenv("HF_TOKEN"))

# Initiate the chat
chat_res = user_proxy.initiate_chat(
    assistant,
    message="What date is today? Compare the year-to-date gain for NVDA and SQQQ.",
)