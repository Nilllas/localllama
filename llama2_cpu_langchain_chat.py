from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_community.llms import HuggingFaceTextGenInference


template_messages = [
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)


# Callbacks support token-wise streaming
callback_manager = CallbackManager(handlers=[StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/home/nilllas/.cache/huggingface/hub/models--hugging-quants--Llama-3.2-3B-Instruct-Q8_0-GGUF/snapshots/7ef7efff7d2c14e5d6161a0c7006e1f2fea6ec79/llama-3.2-3b-instruct-q8_0.gguf", #llama-2-7b-chat.Q2_K.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

model = Llama2Chat(llm=llm, callback_manager=callback_manager)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


#llm_chain = model | prompt_template | memory 
llm_chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)


print("Chatbot initialized, ready to chat...")
while True:
    text = input("> ")
    answer = llm_chain.invoke(text)
    print(answer, '\n')