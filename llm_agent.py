from langchain.agents.agent import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.schema.messages import SystemMessage

from config import OPENAI_API_KEY
from suggestion_prompt import FAQ_prompt
from utils import custom_google_search_tool

Tools = [
    custom_google_search_tool
]

class OpenAIChatAgent:
    system_message = SystemMessage(
        content='You are a helpful AI assistent'
    )
    
    extra_prompt_message = [
        SystemMessagePromptTemplate.from_template(
            """\n---CHAT HISTORY: {chat_history}---\n
            \n---PDF uploaded: {text}---\n"""
        )
    ]

    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo-16k",
            temperature=0.2,
            max_tokens=1000
        )
        print('????========', self.llm.model_name ,' ================')
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=OpenAIFunctionsAgent.from_llm_and_tools(
                self.llm,
                Tools,
                system_message=self.system_message,
                extra_prompt_messages=self.extra_prompt_message
            ),
            tools=Tools
        )
    
    def run(self, chat_history, human_input, st_gen, text):
         st_callback = StreamlitCallbackHandler(st_gen.container())
         chat_history = [f"{message['role']}: {message['content']}" for message in chat_history]

         ai_response = self.agent_executor.run(
             chat_history=str(chat_history),
             text = str(text),
             input=human_input,
             callbacks=[st_callback],
             verbose=True
         )
         return ai_response

class GenFAQsLLM:
    def __init__(self, llm_temp: float = 1.0):
        self.llm_temp = llm_temp
        self.faq_prompt_template = FAQ_prompt
        self.llm = OpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=self.llm_temp,
            temperature=0.2,
            max_tokens=1000
        )

        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self.faq_prompt_template,
            verbose = True,
        )
    
    def run(self, chat_history, language, text,  n_faqs=4):
        chat_history = [f"{message['role']}: {message['content']}" for message in chat_history]
        input_dict = {
            "num": n_faqs,
            "language": language,
            "text": text,
        }
        input_list = [input_dict for _ in range(n_faqs)]

        faqs = self.llm_chain.apply(input_list)

        return [faq['text'] for faq in faqs]