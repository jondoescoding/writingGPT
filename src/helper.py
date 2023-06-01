# Python 
import os
from dataclasses import dataclass
from typing import List
# LangChain
from langchain import LLMChain, llms
from langchain import PromptTemplate

def chain(llm:llms, template:str, inputVariables: List[str], output_key:str):
        """Create a LLM Chain

        Args:
            llm (llms): One of Langchain's integration for LLMs
            template (str): what will be passed to the LLMs
            inputVariables (List[str]): The input for the chain
            output_key (str): the name of the final output from the chain

        Returns:
            _type_: A LLMChain
        """
        return LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=inputVariables,
                template=template
            ),
            output_key=output_key
        )