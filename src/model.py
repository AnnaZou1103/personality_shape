import openai
from openai import OpenAI
from utils.krippendorff_alpha import krippendorff
from sklearn.metrics import cohen_kappa_score
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import anthropic

from utils.utils import post_process


class LLM:
    def __init__(self, api_key, model, model_name, voter):
        """
        Initialize the RaLLM class with the OpenAI API key.

        Parameter:
        - api_key (str): Your OpenAI or Claude API key.
        """
        self.model_type = model
        if model == 'gpt':
            self.client = OpenAI(api_key=api_key)
        elif model == 'claude':
            self.client = anthropic.Anthropic(api_key=api_key)

        self.model_name = model_name
        self.voter = voter

    # this decorator is used to retry if the rate limits are exceeded
    @retry(
        reraise=True,
        stop=stop_after_attempt(1000),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(retry_if_exception_type(openai.APITimeoutError)
               | retry_if_exception_type(openai.APIError)
               | retry_if_exception_type(openai.APIConnectionError)
               | retry_if_exception_type(openai.RateLimitError)),
    )
    def get_response(self, complete_prompt):
        """
        This function uses the OpenAI API to code a given sentence based on the provided complete_prompt.

        Parameters:
        - complete_prompt (str): The generated natural language prompt to be sent to the LLM.
        - engine (str, optional, default="gpt-3.5-turbo"): The OpenAI engine to be used for response generation.

        Returns:
        - list of str: The responses from the OpenAI API.
        """
        # See API document at https://beta.openai.com/docs/api-reference/completions/create
        # max tokens: 100 is enough for single question. 
        # temperature: 0 for greedy (argmax).
        if self.model_type == 'gpt':
            response = self.client.chat.completions.create(
                model=self.model_name, #'gpt-3.5-turbo'
                max_tokens=1000,
                messages=[{"role": "user", "content": complete_prompt}],
                temperature=0.0,
                n=self.voter
            )
            return [response.choices[i].message.content for i in range(len(response.choices))]
        elif self.model_type == 'claude':
            response = self.client.messages.create(
                model=self.model_name, #"claude-3-opus-20240229, claude-3-haiku-20240307"
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": complete_prompt}
                ],
                temperature=0.0,
            )
            return [response.content[i].text for i in range(len(response.content))]

    def llm_translator(self, data, language1, language2, engine="davinci-002"):
        """
        This function uses the OpenAI API to translate a given sentence from one language to another.

        Parameters:
        - data (str): The input sentence to be translated.
        - language1 (str): The source language of the input sentence.
        - language2 (str): The target language to translate the input sentence to.
        - engine (str, optional, default="text-davinci-003"): The OpenAI engine to be used for translation. Defaults to "text-davinci-003".

        Returns:
        - str: The translated sentence.
        """
        instruction = """\
                translate the following sentence from """ + language1 + """ to """ + language2 + """.
                """

        # See API document at https://beta.openai.com/docs/api-reference/completions/create
        # max tokens: 100 is enough for single question. 
        # temperature: 0 for greedy (argmax). 
        response = self.client.chat.completions.create(model=engine,
                                                       messages=[{"role": "user", "content": "\n".join(
                                                           [instruction, 'Sentence: ' + data])}],
                                                       max_tokens=100, temperature=0.0)
        response = response["choices"][0]["text"].strip()
        return response

    def cohens_kappa_measure(self, code, result):
        """
        This function is used to calculate the Cohen's Kappa measure.

        Parameters:
        - code (list): A list of coded items assigned by the first coder.
        - result (list): A list of coded items assigned by the second coder.

        Returns:
        - float: The Cohen's Kappa measure, a value between -1 and 1 representing the level of agreement between the two coders.
        """
        return cohen_kappa_score(code, result)

    def krippendorff_alpha_measure(self, code, result, code_set):
        """
        This function is used to calculate the Krippendorff's alpha measure.

        Parameters:
        - code (list): A list of coded items assigned by the first coder.
        - result (list): A list of coded items assigned by the second coder.
        - codeset (list): A list of codes from the codebook.

        Returns:
        - float: The Krippendorff's alpha measure, a value between -1 and 1 representing the level of agreement between the two coders.
        """
        code_converted = [code_set.index(i) for i in code]
        result_converted = [code_set.index(i) for i in result]
        return krippendorff.krippendorff([code_converted, result_converted], missing_items='')
