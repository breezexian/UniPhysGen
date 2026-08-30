import numpy as np
import os
import random
import base64
from openai import OpenAI

random.seed(123)  # Set random seed to 123

def Qwen(image_path, prompt):
    # Base64 encoding of the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image(image_path)
    client = OpenAI(
        # If the environment variable is not configured, replace the line below with your API Key: api_key="sk-xxx"
        api_key='your_api_key',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # Using f-string to create a string containing the BASE64 encoded image data.
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def GPT4V(image_path, prompt):
    client = OpenAI(api_key="your_api_key")

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0]


def get_image_files(directory):
    image_files = []
    # Iterate over the directory
    for i in range(1, 2):  # Modify range if more folders need to be processed
        sub_path = str(i).zfill(2)
        now_path = os.path.join(directory, sub_path)
        all_png = os.listdir(now_path)
        for png in all_png:
            img_path = os.path.join(now_path, png)
            image_files.append(img_path)

    image_files = sorted(image_files, key=lambda x: (x.split('/')[3], x.split('/')[4]))

    return image_files




