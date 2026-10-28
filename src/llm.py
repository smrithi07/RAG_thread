import time

from google import genai
from google.genai import types

import config

client = None


def get_client():
    global client
    if client is None:
        if config.GEMINI_API_KEY == "":
            raise RuntimeError("GEMINI_API_KEY missing: copy .env.example to .env and add your key")
        client = genai.Client(api_key=config.GEMINI_API_KEY)
    return client


def call_gemini(prompt, json_mode=False):
    if json_mode:
        mime = "application/json"
    else:
        mime = "text/plain"
    settings = types.GenerateContentConfig(temperature=config.GEMINI_TEMPERATURE, response_mime_type=mime)
    # outside the retry loop so a missing key fails at once
    c = get_client()

    waits = [5, 10, 20]
    for attempt in range(len(waits) + 1):
        try:
            response = c.models.generate_content(model=config.GEMINI_MODEL, contents=prompt,
                                                            config=settings)
            return response.text
        except Exception as e:
            if attempt == len(waits):
                raise
            print("  Gemini error, retrying in", waits[attempt], "s:", e)
            time.sleep(waits[attempt])
