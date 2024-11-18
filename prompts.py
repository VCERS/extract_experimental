#!/usr/bin/python3

from typing import Any, Union
from transformers import AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, PromptValue

# NOTE: langchain-core >= 0.3.15

class HFChatPromptValue(ChatPromptValue):
  tokenizer: Any = None
  def to_string(self) -> str:
    hf_messages = []
    for m in self.messages:
      if isinstance(m, HumanMessage):
        role = 'user'
      elif isinstance(m, AIMessage):
        role = 'assistant'
      elif isinstance(m, SystemMessage):
        role = 'system'
      else:
        raise Exception(f'Got unsupported message type: {m}')
      hf_messages.append({'role': role, 'content': m.content})
    return self.tokenizer.apply_chat_template(hf_messages, tokenize = False, add_generation_prompt = True)

class HFChatPromptTemplate(ChatPromptTemplate):
  tokenizer: Any = None
  def format_prompt(self, **kwargs: Any) -> PromptValue:
    messages = self.format_messages(**kwargs)
    return HFChatPromptValue(messages = messages, tokenizer = self.tokenizer)
  async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
    messages = await self.format_messages(**kwargs)
    return HFChatPromptValue(messages = messages, tokenizer = self.tokenizer)

def experimental_template(tokenizer):
  messages = [
    ('user', '''Given a full text of a paper. Please return the original text of the experimental part of the paper. If the experimental part is not explicitly described in a certain paragraph, summarize the experimental process based on the full text content.

full text of a paper:

{text}''')
  ]
  prompt = HFChatPromptTemplate(messages, tokenizer = tokenizer)
  return prompt

