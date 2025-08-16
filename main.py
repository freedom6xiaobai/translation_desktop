import asyncio
import sys
import os
import threading
import time
from collections import deque

import sounddevice as sd
import numpy as np
import wave
import datetime
import whisperx
import tempfile
import soundfile as sf

from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QVBoxLayout, QScrollArea, QPushButton, \
    QFileDialog, \
    QMessageBox, QHBoxLayout, QMenu
from PySide6.QtGui import QCursor
from PySide6.QtCore import Qt, QCoreApplication, QTimer, QMetaObject, Q_ARG
from PySide6.QtCore import Slot
# from googletrans import Translator
# from deep_translator import GoogleTranslator
from azure_ts import translate_text_azure
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import warnings

warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends.*")

SILENCE_THRESHOLD = 0.01  # 你可根据麦克风灵敏度调整
silence_duration_limit = 3  # 静音持续3秒才认定为静音


# 主mainwindowffmpeg
class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.resize(800, 600)
        self.setWindowTitle('实时翻译Translate')
        self.setAcceptDrops(True)
        self.setAutoFillBackground(True)
        self.setWindowIcon(QIcon('./resource/icon.png'))
        # self.setCursor(QCursor(Qt.WaitCursor))  # 设置为沙漏等待光标

        # # 创建动作
        # action_hello = QAction("说你好", self)
        # action_hello.triggered.connect(self.say_hello)
        # self.addAction(action_hello)
        #
        # action_hello1 = QAction("说你好1", self)
        # action_hello1.triggered.connect(self.say_hello1)
        # self.addAction(action_hello1)
        #
        # action_exit = QAction("退出", self)
        # action_exit.triggered.connect(self.close)
        # self.addAction(action_exit)

    # def say_hello(self):
    #     # 创建文件对话框
    #     file_dialog = QFileDialog(self)
    #
    #     # 设置文件对话框的标题
    #     file_dialog.setWindowTitle("Choose a File")
    #
    #     # 设置对话框模式为打开文件
    #     file_dialog.setFileMode(QFileDialog.ExistingFile)
    #
    #     # 显示文件对话框，并获取用户选择的文件路径
    #     selected_file, _ = file_dialog.getOpenFileName(self, "Open File", "", "All Files (*);;Text Files (*.txt)")
    #
    #     # 如果用户选择了文件，将文件路径输出到控制台
    #     if selected_file:
    #         print(f"Selected file: {selected_file}")
    #
    # def say_hello1(self):
    #     # 创建询问消息框
    #     result = QMessageBox.question(self, "Question", "Do you want to proceed?", QMessageBox.Yes | QMessageBox.No)
    #
    #     if result == QMessageBox.Yes:
    #         print("User clicked Yes.")
    #     else:
    #         print("User clicked No.")


class MyWidget(QWidget):
    @Slot()
    def call_update_ui(self):
        self.update_ui(self._pending_result_text)

    def __init__(self, _):
        super().__init__()
        # 静音判断
        self.silence_counter = 0
        self.executor = ThreadPoolExecutor(max_workers=2)

        # 初始化参数
        self.recording = False
        self.fs = 44100  # 采样率
        self.recording_frames = []
        self.record_thread = None
        self.last_audio_chunk = None  # 用于存储最后一段音频数据

        # whisperx
        self.audio_buffer = deque()
        self.last_trans_time = time.time()
        self.model = whisperx.load_model("small", device="cpu", compute_type="int8")
        print("模型内容：", self.model)

        # 逻辑处理线程
        self.stop_transcribe = threading.Event()
        self.transcribe_thread = None
        self.buffer_lock = threading.Lock()

        # 缓存识别文本
        self.transcribe_cache = []
        self.cache_max_len = 5

        # 计时器
        self.record_seconds = 0
        self.record_timer = QTimer()
        self.record_timer.setInterval(1000)
        self.record_timer.timeout.connect(self.update_record_time)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # 创建消息区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            padding: 8px;
            border: 1px solid #b2ebf2;
        """)
        self.msg_label = QLabel()
        self.msg_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.msg_label.setStyleSheet("border: none;")
        self.msg_label.setWordWrap(True)
        self.msg_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.msg_label.customContextMenuRequested.connect(self.show_msg_label_menu)
        self.scroll_area.setWidget(self.msg_label)
        layout.addWidget(self.scroll_area)

        # 创建翻译区域
        # self.scroll_area_bottom = QScrollArea()
        # self.scroll_area_bottom.setWidgetResizable(True)
        # self.scroll_area_bottom.setStyleSheet("""
        #     padding: 8px;
        #     border: 1px solid #DDFF00;
        # """)
        # self.msg_label_bottom = QLabel()
        # self.msg_label_bottom.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # self.msg_label_bottom.setStyleSheet("border: none;")
        # self.msg_label_bottom.setWordWrap(True)
        # self.msg_label_bottom.setContextMenuPolicy(Qt.CustomContextMenu)
        # self.msg_label_bottom.customContextMenuRequested.connect(self.show_msg_label_bottom_menu)
        # self.scroll_area_bottom.setWidget(self.msg_label_bottom)
        # layout.addWidget(self.scroll_area_bottom)

        # 创建底部的录制按钮
        tool_layout = QHBoxLayout()
        tool_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        # 录制按钮
        self.record_button = QPushButton()
        self.record_button.setStyleSheet("font-size: 16px; color: #FFFFFF")
        self.record_button.setFixedSize(80, 50)
        self.record_button.setText("▶")  # 默认状态为播放三角
        self.record_button.clicked.connect(self.start_recording)
        tool_layout.addWidget(self.record_button)

        # 进度
        self.record_time_label = QLabel("00:00:00")
        self.record_time_label.setStyleSheet("font-size: 16px; color: #FFFFFF")
        self.record_time_label.setFixedSize(80, 50)
        tool_layout.addWidget(self.record_time_label)

        # 清空按钮
        self.clear_button = QPushButton("Clear All")
        self.clear_button.setStyleSheet("font-size: 16px; color: #FFFFFF")
        self.clear_button.setFixedSize(80, 50)
        self.clear_button.clicked.connect(self.clear_text)
        tool_layout.addWidget(self.clear_button)

        layout.addLayout(tool_layout)

    # msg 菜单
    def show_msg_label_menu(self, pos):
        menu = QMenu()
        copy_all_action = QAction("All Copy", self)
        copy_all_action.triggered.connect(self.copy_all_msg_label_text)
        menu.addAction(copy_all_action)

        # 在 msg_label 上弹出菜单，pos 是局部坐标
        menu.exec(self.msg_label.mapToGlobal(pos))

    def copy_all_msg_label_text(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.msg_label.text())

    def show_msg_label_bottom_menu(self, pos):
        menu = QMenu()
        copy_all_action = QAction("All Copy", self)
        copy_all_action.triggered.connect(self.copy_all_msg_label_bottom_text)
        menu.addAction(copy_all_action)

        # 在 msg_label 上弹出菜单，pos 是局部坐标
        menu.exec(self.msg_label_bottom.mapToGlobal(pos))

    # msg translate 菜单

    def copy_all_msg_label_bottom_text(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.msg_label_bottom.text())

    # clear
    def clear_text(self):
        self.msg_label.setText("")
        # self.msg_label_bottom.setText("")

    # 计时器更新
    def update_record_time(self):
        self.record_seconds += 1
        hours = self.record_seconds // 3600
        minutes = (self.record_seconds % 3600) // 60
        seconds = self.record_seconds % 60
        self.record_time_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    # 静音检测函数
    def detect_silence(self, audio_chunk):
        # 简单能量法检测静音
        if audio_chunk is None or len(audio_chunk) == 0:
            return True
        energy = np.mean(np.abs(audio_chunk))
        return energy < SILENCE_THRESHOLD

    # 文本相似度判定（简单版：Jaccard/重叠度）
    def is_similar(self, text1, text2):
        # 只要有较大重叠就认为重复
        if not text1 or not text2:
            return False
        set1 = set(text1.strip())
        set2 = set(text2.strip())
        if not set1 or not set2:
            return False
        overlap = len(set1 & set2) / max(len(set1), 1)
        return overlap > 0.8

    # 合并缓存文本
    def merge_cache_text(self):
        # 用 set 去重 + 保留顺序
        seen = set()
        merged = []
        for text in self.transcribe_cache:
            if text not in seen:
                seen.add(text)
                merged.append(text)
        return "\n".join(merged)

    # 分句函数：按英文标点分句
    def split_into_sentences(self, text):
        import re
        # 按英文句号、问号、感叹号拆分
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text.strip())
        # 去除空句
        return [s.strip() for s in sentences if s.strip()]

    # 合并唯一且最长表达的句子，去除被包含或重复的句子
    def merge_unique_sentences(self, sentences):
        # 保留唯一且最长表达的句子
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

    # 增加临时缓存
    def transcribe_audio_chunk(self, audio_chunk):
        tmp_path = "debug_audio.wav"

        # 检查当前片段是否过短
        if len(audio_chunk) / self.fs < 2.0 or len(audio_chunk) < self.fs * 2.0:
            if self.last_audio_chunk is not None:
                print("⚠️ 当前片段过短，拼接上一段音频增强上下文")
                audio_chunk = np.concatenate([self.last_audio_chunk, audio_chunk], axis=0)

        self.last_audio_chunk = audio_chunk  # 缓存当前的

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
            # --- 句子合并逻辑 ---
            sentences = self.split_into_sentences(result_text)
            merged_sentences = self.merge_unique_sentences(sentences)
            result_text = " ".join(merged_sentences)
        except Exception as e:
            print(f"[转写错误]: {e}")
            result_text = f"[识别失败]: {e}"
        return result_text

    # 线程池 识别speaker 内容 实现实时录些
    def transcribe_loop(self):
        silence_count = 0
        silence_limit = silence_duration_limit  # 秒
        silence_chunk_samples = int(self.fs * 0.5)  # 0.5秒为单位检测
        buffer_chunks = []
        buffer_duration = 0
        while not self.stop_transcribe.is_set():
            with self.buffer_lock:
                if len(self.audio_buffer) == 0:
                    time.sleep(0.1)
                    continue

                chunks = []
                total_duration = 0
                # 收集音频片段，直到累计约0.5秒
                while self.audio_buffer and total_duration < silence_chunk_samples:
                    chunk = self.audio_buffer.popleft()
                    chunks.append(chunk)
                    total_duration += len(chunk)

            if not chunks:
                time.sleep(0.2)
                continue

            audio_chunk = np.concatenate(chunks, axis=0)
            audio_chunk = audio_chunk.flatten().astype(np.float32)
            buffer_chunks.append(audio_chunk)
            buffer_duration += len(audio_chunk)

            # 检查静音
            if self.detect_silence(audio_chunk):
                silence_count += 0.5
            else:
                silence_count = 0

            # 如果达到缓存上限或者检测到静音持续足够时间，则触发识别
            if buffer_duration >= self.fs * 5.0 or silence_count >= silence_limit:
                # 合并缓存
                full_audio = np.concatenate(buffer_chunks, axis=0)
                full_audio = full_audio.flatten().astype(np.float32)
                result_text = self.transcribe_audio_chunk(full_audio)
                # 判重
                if self.transcribe_cache:
                    last_text = self.transcribe_cache[-1]
                    if self.is_similar(last_text, result_text):
                        # 跳过重复
                        buffer_chunks.clear()
                        buffer_duration = 0
                        silence_count = 0
                        continue
                # 缓存并合并
                self.transcribe_cache.append(result_text)
                if len(self.transcribe_cache) > self.cache_max_len:
                    self.transcribe_cache = self.transcribe_cache[-self.cache_max_len:]
                merged_text = self.merge_cache_text()
                print("识别结果（缓存合并）:", merged_text)
                self._pending_result_text = merged_text
                QMetaObject.invokeMethod(self, "call_update_ui", Qt.QueuedConnection)
                # 清空缓存
                buffer_chunks.clear()
                buffer_duration = 0
                silence_count = 0

    # 音频转换文本
    def start_transcribe_thread(self, indata=None):
        print("启动转录线程")
        self.stop_transcribe.clear()
        if not self.transcribe_thread or not self.transcribe_thread.is_alive():
            self.transcribe_thread = threading.Thread(target=self.transcribe_loop)
            self.transcribe_thread.daemon = True
            self.transcribe_thread.start()

    # 录制音频
    def record_audio(self):
        self.recording_frames = []

        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            if self.recording:
                with self.buffer_lock:
                    self.audio_buffer.append(indata.copy())
                self.recording_frames.append(indata.copy())
            else:
                raise sd.CallbackStop()

        with sd.InputStream(samplerate=self.fs, channels=1, callback=callback):
            while self.recording:
                sd.sleep(100)

        # 录音结束后保存为 wav 文件
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"cache/recorded_audio_{timestamp}.wav"
        wav_file = wave.open(filename, 'wb')
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16bit
        wav_file.setframerate(self.fs)

        # 拼接所有录音数据，并转换为 int16
        audio_data = np.concatenate(self.recording_frames)
        # 归一化处理：避免溢出 + 保持动态范围
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val  # 归一化到 -1.0 ~ 1.0
        GAIN = 3.0  # 可以调成 2.0 到 5.0 之间
        audio_data *= GAIN
        audio_data = np.clip(audio_data, -1.0, 1.0)  # 避免溢出
        # 转为 16bit PCM
        audio_data_int16 = (audio_data * 32767).astype(np.int16)
        wav_file.writeframes(audio_data_int16.tobytes())
        print(audio_data.dtype)
        wav_file.close()
        print(f"录音已保存到 {filename}")

    # 录制点击
    def start_recording(self):
        if self.recording:
            self.stop_transcribe.set()
            self.recording = False
            self.record_button.setStyleSheet("font-size: 16px; color: #FFFFFF")
            self.record_button.setText("▶")  # 回到初始状态
            self.record_timer.stop()
            self.record_seconds = 0
            self.record_time_label.setText("00:00:00")
            if self.record_thread:
                self.record_thread.join()
                self.record_thread = None

            if self.transcribe_thread:
                self.transcribe_thread.join()
                self.transcribe_thread = None

        # 录音结束时保存 msg_label 文本到本地文件
        transcript_text = self.msg_label.text()
        if transcript_text.strip():  # 非空时保存
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            filename = f"text/transcript_{timestamp}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            print(f"识别文本已保存到 {filename}")

        else:
            # ✅ 添加这行来启动转录线程
            self.start_transcribe_thread()

            self.recording = True
            self.record_button.setStyleSheet("font-size: 32px; color: #FF0000")
            self.record_button.setText("⏺")  # 录制中，用红色圆点
            self.record_timer.start()
            self.record_thread = threading.Thread(target=self.record_audio)
            self.record_thread.start()

    # translate 翻译
    def exchange_cn(self, text):
        # 在后台线程中执行翻译
        def do_translate(text):
            try:
                translated = translate_text_azure("hello world")
                # translated = GoogleTranslator(source='auto', target='zh-CN').translate(text)
                return translated
                # translator = Translator()
                # result = translator.translate(text, src='en', dest='zh-CN')
                # return result.text
            except Exception as e:
                return f"[翻译失败]: {e}"

        future = self.executor.submit(do_translate, text)
        future.add_done_callback(partial(self.on_translation_finished, original=text))

    def on_translation_finished(self, future, original):
        translated_text = future.result()
        print(translated_text)
        self.msg_label_bottom.setText(self.msg_label_bottom.text() + ' ' + translated_text)
        scroll_bar = self.scroll_area_bottom.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    # 刷新ui
    def update_ui(self, result_text):
        # print("[UI] 更新调用中")
        self.msg_label.setText(self.msg_label.text() + ' ' + result_text)
        scroll_bar = self.scroll_area.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
        # asyncio.run(self.exchange_cn(result_text))
        # self.exchange_cn(result_text)


if __name__ == '__main__':
    print("Hello from PyInstaller!")
    print('~/.cache/whisper')
    print(whisperx.__file__)  # 检查路径
    print(dir(whisperx))  # 应该有 load_model
    os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin/"  # 替换成你的ffmpeg安装目录
    print("PATH=", os.environ.get("PATH"))
    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()

    # ✅ 创建中央控件区域
    widget = MyWidget(mainWindow)
    mainWindow.setCentralWidget(widget)

    mainWindow.show()
    app.exec()
