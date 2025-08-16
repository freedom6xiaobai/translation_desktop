import numpy as np
import soundfile as sf
import time
from collections import deque

from PySide6.QtCore import QMetaObject, Qt, QGenericArgument
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RealTimeTranscriber:
    def __init__(self, model, fs=44100, cache_max_len=50, silence_duration_limit=2.0):
        self.model = model
        self.fs = fs
        self.cache_max_len = cache_max_len
        self.silence_duration_limit = silence_duration_limit

        # self.audio_buffer = deque()
        # self.buffer_lock = None  # 你原来线程锁对象
        # self.stop_transcribe = None  # 线程停止事件
        self.transcribe_cache = []
        self.last_audio_chunk = None
        self._pending_result_text = ""

        # 句向量模型，用于文本相似度计算
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print('~/.cache/huggingface/transformers/')

    # 静音检测：基于能量阈值
    def detect_silence(self, audio_chunk, energy_threshold=1e-4):
        energy = np.mean(audio_chunk ** 2)
        return energy < energy_threshold

    # 文本相似度判定
    def is_similar(self, text1, text2, threshold=0.85):
        if not text1 or not text2:
            return False
        vecs = self.sbert_model.encode([text1, text2])
        sim = cosine_similarity([vecs[0]], [vecs[1]])[0][0]
        return sim >= threshold

    # 合并缓存文本，去除相似文本
    def merge_cache_text(self):
        merged = []
        for text in self.transcribe_cache:
            if not any(self.is_similar(text, m) for m in merged):
                merged.append(text)
        return "\n".join(merged)

    def split_into_sentences(self, text):
        import re
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def merge_unique_sentences(self, sentences):
        result = []
        for i, s in enumerate(sentences):
            is_contained = False
            for j, t in enumerate(sentences):
                if i != j and s in t:
                    is_contained = True
                    break
            if not is_contained and s not in result:
                result.append(s)
        return result

    def transcribe_audio_chunk(self, audio_chunk):
        tmp_path = "debug_audio.wav"
        # 片段过短拼接上一段
        if len(audio_chunk) / self.fs < 1.0 and self.last_audio_chunk is not None:
            audio_chunk = np.concatenate([self.last_audio_chunk, audio_chunk], axis=0)
        self.last_audio_chunk = audio_chunk

        sf.write(tmp_path, audio_chunk, self.fs)
        try:
            result = self.model.transcribe(tmp_path)
            segments = result.get("segments", [])
            result_text = ""
            if segments:
                for segment in segments:
                    speaker = segment.get("speaker")
                    text = segment.get("text", "")
                    if speaker:
                        result_text += f"[{speaker}] {text}\n"
                    else:
                        result_text += f"{text}\n"
            else:
                result_text = result.get("text", "")

            sentences = self.split_into_sentences(result_text)
            merged_sentences = self.merge_unique_sentences(sentences)
            result_text = " ".join(merged_sentences)
        except Exception as e:
            print(f"[转写错误]: {e}")
            result_text = f"[识别失败]: {e}"
        return result_text


    def transcribe_loop(self, audio_buffer, stop_event, buffer_lock, ui_callback):
        silence_count = 0.0
        silence_limit = self.silence_duration_limit
        silence_chunk_samples = int(self.fs * 0.2)  # 0.2秒检测
        buffer_chunks = []
        buffer_duration = 0
        while not stop_event.is_set():
            with buffer_lock:
                if len(audio_buffer) == 0:
                    time.sleep(0.2)
                    continue

                chunks = []
                total_duration = 0
                while audio_buffer and total_duration < silence_chunk_samples:
                    chunk = audio_buffer.popleft()
                    chunks.append(chunk)
                    total_duration += len(chunk)

            if not chunks:
                time.sleep(0.2)
                continue

            audio_chunk = np.concatenate(chunks, axis=0).astype(np.float32).flatten()
            buffer_chunks.append(audio_chunk)
            buffer_duration += len(audio_chunk)

            if self.detect_silence(audio_chunk):
                silence_count += 0.5
            else:
                silence_count = 0.0

            # 触发条件：缓存时间长或静音超过限制
            if buffer_duration >= self.fs * 10.0 or silence_count >= silence_limit:
                full_audio = np.concatenate(buffer_chunks, axis=0).astype(np.float32).flatten()
                result_text = self.transcribe_audio_chunk(full_audio)

                # 判重
                if self.transcribe_cache:
                    last_text = self.transcribe_cache[-1]
                    if self.is_similar(last_text, result_text):
                        # 跳过重复，清空缓存
                        buffer_chunks.clear()
                        buffer_duration = 0
                        silence_count = 0
                        continue

                self.transcribe_cache.append(result_text)
                if len(self.transcribe_cache) > self.cache_max_len:
                    self.transcribe_cache = self.transcribe_cache[-self.cache_max_len:]

                merged_text = self.merge_cache_text()
                print("识别结果（缓存合并）:", merged_text)
                self._pending_result_text = merged_text

                # UI 更新方法，需你自己实现
                # QMetaObject.invokeMethod(ui_callback, "call_update_ui", Qt.QueuedConnection)
                # QMetaObject.invokeMethod(ui_callback, b"call_update_ui", Qt.QueuedConnection)
                ui_callback.update_ui_signal.emit(self._pending_result_text)

                buffer_chunks.clear()
                buffer_duration = 0
                silence_count = 0
