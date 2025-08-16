import asyncio
import os
import whisper
from googletrans import Translator


async def main():
    translator = Translator()
    result = await translator.translate("Hello, how are you?", src='en', dest='zh-cn')
    print(result.text)  # 输出：你好，你好吗？

if __name__ == '__main__':
    # os.environ["PATH"] += os.pathsep + "/usr/local/bin"  # 替换成你的ffmpeg安装目录
    # print("PATH=", os.environ.get("PATH"))
    #
    # model = whisper.load_model("base")
    # result = model.transcribe("recorded_audio_2025_05_16_19_26_09.wav")
    # print(result["text"])

    asyncio.run(main())