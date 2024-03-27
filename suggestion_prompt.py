from langchain.prompts.chat import (SystemMessagePromptTemplate, AIMessagePromptTemplate,
                                    HumanMessagePromptTemplate, ChatPromptTemplate)


prompt_template = """
You are doing Shareholder activism. Please write a suggestion letter for general meetings for shareholders in formal way.

# 안건
Question : question
Choice : choice
Answer : answer

---
Question Info
Types: 
Language: {language}
---

TEXT:
{text}
"""

FAQ_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt_template)
])
