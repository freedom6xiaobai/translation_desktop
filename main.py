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
import whisper
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
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
from functools import partial

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

        # 加载模型
        self.audio_buffer = deque()
        self.last_trans_time = time.time()
        self.model = whisper.load_model("small")  # 模型初始化建议单次完成

        # 逻辑处理线程
        self.stop_transcribe = threading.Event()
        self.transcribe_thread = None
        self.buffer_lock = threading.Lock()

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
        self.scroll_area_bottom = QScrollArea()
        self.scroll_area_bottom.setWidgetResizable(True)
        self.scroll_area_bottom.setStyleSheet("""
            padding: 8px;
            border: 1px solid #DDFF00;
        """)
        self.msg_label_bottom = QLabel()
        self.msg_label_bottom.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.msg_label_bottom.setStyleSheet("border: none;")
        self.msg_label_bottom.setWordWrap(True)
        self.msg_label_bottom.setContextMenuPolicy(Qt.CustomContextMenu)
        self.msg_label_bottom.customContextMenuRequested.connect(self.show_msg_label_bottom_menu)
        self.scroll_area_bottom.setWidget(self.msg_label_bottom)
        layout.addWidget(self.scroll_area_bottom)

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

    # msg translate 菜单
    def show_msg_label_bottom_menu(self, pos):
        menu = QMenu()
        copy_all_action = QAction("All Copy", self)
        copy_all_action.triggered.connect(self.copy_all_msg_label_bottom_text)
        menu.addAction(copy_all_action)

        # 在 msg_label 上弹出菜单，pos 是局部坐标
        menu.exec(self.msg_label_bottom.mapToGlobal(pos))

    def copy_all_msg_label_bottom_text(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.msg_label_bottom.text())

    # clear
    def clear_text(self):
        self.msg_label.setText("")
        self.msg_label_bottom.setText("")

    # 计时器更新
    def update_record_time(self):
        self.record_seconds += 1
        hours = self.record_seconds // 3600
        minutes = (self.record_seconds % 3600) // 60
        seconds = self.record_seconds % 60
        self.record_time_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    # 增加临时缓存
    def transcribe_audio_chunk(self, audio_chunk):
        tmp_path = "debug_audio.wav"
        sf.write(tmp_path, audio_chunk, self.fs)
        try:
            result = self.model.transcribe(tmp_path, fp16=False, language='en')
            result_text = result.get('text', '')
        except Exception as e:
            result_text = f"[识别失败]: {e}"
        return result_text

    # 线程池
    def transcribe_loop(self):
        while not self.stop_transcribe.is_set():
            # print("循环体运行中...")
            time.sleep(1.5)
            with self.buffer_lock:
                # print(f"当前音频缓冲区大小：{len(self.audio_buffer)}")
                if len(self.audio_buffer) == 0:
                    time.sleep(0.1)
                    continue
                chunks = list(self.audio_buffer)
                self.audio_buffer.clear()
                audio_chunk = np.concatenate(chunks, axis=0)
                duration = len(audio_chunk) / self.fs
                # print(f"拼接音频长度: {len(audio_chunk)}, 时长: {duration:.2f}s")
            audio_chunk = audio_chunk.flatten().astype(np.float32)
            result_text = self.transcribe_audio_chunk(audio_chunk)
            print("识别结果：", result_text)

            self._pending_result_text = result_text
            QMetaObject.invokeMethod(self, "call_update_ui", Qt.QueuedConnection)

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
                translated = GoogleTranslator(source='auto', target='zh-CN').translate(text)
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
    # print(whisper.__file__)          # 检查路径
    # print(dir(whisper))              # 应该有 load_model
    os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin/"  # 替换成你的ffmpeg安装目录
    print("PATH=", os.environ.get("PATH"))
    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()

    # ✅ 创建中央控件区域
    widget = MyWidget(mainWindow)
    mainWindow.setCentralWidget(widget)

    mainWindow.show()
    app.exec()
