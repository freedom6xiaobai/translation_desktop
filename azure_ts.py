import uuid
import requests

AZURE_TRANSLATOR_KEY = "你的 Azure Translator 密钥"
AZURE_REGION = "australiaeast"  # 或你创建时选择的区域
AZURE_ENDPOINT = "https://api.cognitive.microsofttranslator.com/"  # 不带路径

def translate_text_azure(text, to_lang="zh-Hans", from_lang="en"):
    path = "/translate?api-version=3.0"
    params = f"&from={from_lang}&to={to_lang}"
    url = AZURE_ENDPOINT + path + params

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_REGION,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4())
    }

    body = [{"text": text}]

    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        result = response.json()
        return result[0]["translations"][0]["text"]
    except Exception as e:
        print("翻译失败:", e)
        return text  # 翻译失败时返回原文