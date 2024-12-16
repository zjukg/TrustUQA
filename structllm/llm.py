import openai 
import time
import requests
from openai import OpenAI
import os


class gpt(object):
    def __init__(self, args):
        self.model = args.model
        OpenAI.api_key = args.key
        self.url = args.openai_url
        
        self.client = OpenAI(base_url=self.url, api_key=OpenAI.api_key)
    
    def get_response(self, prompt, flag=0, num=1):
        if self.model=="text-davinci-003" or type(prompt)==str:
            print("text-davinci-003")
            response = OpenAI.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=800,
                temperature=0.5,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=1,
                # stop=stop_sequences,
                logprobs=1,
                n=1,
                best_of=1,
            )
            return response["choices"][0]['text']
        
        else:
            start_time = time.time()
            while True:
                try:
                    if time.time() - start_time > 300:  # 300 seconds = 5 minutes
                        raise TimeoutError("Code execution exceeded 5 minutes")

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=prompt,
                        temperature=1, #1
                        max_tokens=256,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        n=num
                    )
                    if num == 1 and (not flag):
                        return response.choices[0].message.content
                    else:
                        return response.choices
                
                except requests.exceptions.RequestException as e:
                    print(f"Network error: {e}")
                    print("Retrying in 10 seconds...")
                    time.sleep(10)

                except openai.APIError as e:
                    print('OpenAI.APIError\nRetrying...')
                    print(e)
                    time.sleep(20)

                except openai.APIConnectionError as e:
                    print('OpenAI.APIConnectionError\n{e}\nRetrying...')
                    time.sleep(20)

                except openai.RateLimitError as e:
                    err_mes = str(e)
                    if "You exceeded your current quota" in err_mes:
                        print("You exceeded your current quota")
                    print('OpenAI.error.RateLimitError\nRetrying...')
                    time.sleep(30)

                except openai.APITimeoutError:
                    print('OpenAI.APITimeoutError\nRetrying...')
                    time.sleep(20)

                except TimeoutError as e:
                    # Handle the custom TimeoutError exception
                    print(f"Code execution exceeded 5 minutes: {e}")
                    # Optionally, you can re-raise the exception to terminate the script
                    raise e