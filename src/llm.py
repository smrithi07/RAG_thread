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

    models = [config.GEMINI_MODEL]
    if config.GEMINI_FALLBACK_MODEL != "":
        models.append(config.GEMINI_FALLBACK_MODEL)

    last_error = ""
    for m in range(len(models)):
        for attempt in range(2):
            try:
                response = c.models.generate_content(model=models[m], contents=prompt, config=settings)
                return response.text
            except Exception as e:
                last_error = str(e)
                print("  Gemini " + models[m] + " failed: " + last_error[:80])
                if attempt == 0:
                    time.sleep(5)
    raise RuntimeError("Gemini unavailable (" + last_error[:120] + "). Try again or change GEMINI_MODEL in .env")
