# Translator Core
import os
import time
import threading
import tempfile
from typing import Callable, Optional, Dict, Any, List
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pytorch_i3d import InceptionI3d
from googletrans import Translator
from gtts import gTTS
import pygame

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- Tunables ---
CONF_THRESH = 0.50
ENTER_FRAMES = 1
SUPPRESS_MS = 3000
SMOOTH_WINDOW = 0.6
TOPK_MARGIN = 0.04
MAX_CONTEXT_WORDS = 16
SENTENCE_TRIGGER = 3

class TranslatorContext:
    def __init__(self):
        self.i3d: Optional[nn.Module] = None
        self.wlasl_dict: Dict[int, str] = {}
        self.tokenizer: Optional[AutoTokenizer] = None
        self.k2t_model: Optional[AutoModelForSeq2SeqLM] = None
        self.k2t_params: Dict[str, Any] = {
            "do_sample": True,
            "num_beams": 5,
            "no_repeat_ngram_size": 2,
            "early_stopping": True
        }
        self._run_flag = False
        self._thread: Optional[threading.Thread] = None
        self._ema_probs: Optional[torch.Tensor] = None
        self._consec_ok = 0
        self._last_emitted: Dict[str, float] = {}

        # TTS and translation
        self.tts_language = "English"
        self._tts_lock = threading.Lock()
        self.text_translator = Translator()

    def set_tts_language(self, lang: str):
        """Set TTS language"""
        self.tts_language = lang

    def translate_text(self, text: str, target_lang: str) -> str:
        """Translate text to target language"""
        if not text or not text.strip():
            return ""
        if target_lang == "English":
            return text
        try:
            lang_codes = {"Hindi": "hi", "Spanish": "es"}
            target_code = lang_codes.get(target_lang, "en")
            result = self.text_translator.translate(text, src='en', dest=target_code)
            return result.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def speak(self, text: str):
        """Speak text using Google TTS (better for Hindi/non-Latin scripts)"""
        if not text or not text.strip():
            print("TTS: No text to speak")
            return

        # Translate text first
        translated_text = self.translate_text(text, self.tts_language)

        def _do_speak():
            with self._tts_lock:
                temp_file = None
                try:
                    # Map language to gTTS language codes
                    lang_map = {"English": "en", "Hindi": "hi", "Spanish": "es"}
                    lang_code = lang_map.get(self.tts_language, "en")
                    
                    print(f"TTS: Speaking in {self.tts_language} ({lang_code}): '{translated_text[:50]}...'")
                    
                    # Generate speech audio
                    tts = gTTS(text=translated_text, lang=lang_code, slow=False)
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                        temp_file = fp.name
                        tts.save(temp_file)
                    
                    # Play audio using pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()
                    
                    # Wait for audio to finish
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    
                    pygame.mixer.quit()
                    print("TTS: Finished speaking")
                except Exception as e:
                    print(f"TTS Error: {e}")
                finally:
                    if temp_file and os.path.exists(temp_file):
                        try:
                            os.unlink(temp_file)
                        except:
                            pass

        threading.Thread(target=_do_speak, daemon=True).start()

    def _create_WLASL_dictionary(self, path: str = "preprocess/wlasl_class_list.txt"):
        mapping = {}
        if not os.path.exists(path):
            print(f"Warning: class list file not found: {path}")
            self.wlasl_dict = mapping
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                split_list = line.split()
                if len(split_list) != 2:
                    key = int(split_list[0])
                    value = split_list[1] + " " + split_list[2]
                else:
                    key = int(split_list[0])
                    value = split_list[1]
                mapping[key] = value
        self.wlasl_dict = mapping

    '''def _generate_text_from_keywords(self, keywords: List[str]) -> str:
        input_text = " ".join(keywords)
        if self.tokenizer is None or self.k2t_model is None:
            return input_text
        try:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            output_ids = self.k2t_model.generate(input_ids, **self.k2t_params)
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        except Exception:
            return input_text 
    '''
    def _generate_text_from_keywords(self, keywords: List[str]) -> str:
        """Enhanced sentence generation with ASL grammar handling"""
        if not keywords:
            return ""
        
        # Common ASL verbs that need -ing form
        action_verbs = {
            'paint', 'run', 'eat', 'walk', 'jump', 'read', 'write', 'play', 
            'watch', 'learn', 'swim', 'dance', 'sing', 'cook', 'drive', 'work'
        }
        
        # Try KeyToText first
        input_text = " ".join(keywords)
        
        if self.tokenizer and self.k2t_model:
            try:
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
                output_ids = self.k2t_model.generate(input_ids, **self.k2t_params)
                sentence = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Quality check: if output is too long/weird, use simple rules
                if len(sentence.split()) <= len(keywords) * 2:
                    return sentence
            except Exception:
                pass
        
        # Fallback: Simple ASL grammar rules
        subject = keywords[0]
        verb_idx = None
        
        for i, word in enumerate(keywords[1:], 1):
            if word in action_verbs:
                verb_idx = i
                break
        
        if verb_idx:
            verb = keywords[verb_idx] + "ing"
            objects = keywords[verb_idx+1:] if verb_idx+1 < len(keywords) else []
            obj_str = " a " + " ".join(objects) if objects else ""
            return f"The {subject} is {verb}{obj_str}".strip()
        else:
            return "The " + " ".join(keywords)
    
    def _accept_decision(self, probs: torch.Tensor) -> str:
        topv, topi = torch.topk(probs, 2)
        top1 = float(topv[0])
        top2 = float(topv[1]) if len(topv) > 1 else 0.0
        if top1 < CONF_THRESH or (top1 - top2) < TOPK_MARGIN:
            self._consec_ok = 0
            return " "
        self._consec_ok += 1
        if self._consec_ok < ENTER_FRAMES:
            return " "
        idx = int(topi[0])
        word = self.wlasl_dict.get(idx, " ")
        if word == " ":
            return " "
        now = time.time() * 1000.0
        last = self._last_emitted.get(word, 0.0)
        if (now - last) < SUPPRESS_MS:
            return " "
        self._last_emitted[word] = now
        self._consec_ok = 0
        return word

    def _run_on_tensor(self, clip_cthw: torch.Tensor) -> str:
        clip_cthw = clip_cthw[None, :]
        t = clip_cthw.shape[2]
        clip_cthw = clip_cthw.cuda(non_blocking=True)
        with torch.no_grad():
            per_frame_logits = self.i3d(clip_cthw)
            preds = F.interpolate(per_frame_logits, size=t, mode="linear", align_corners=False)
            preds = preds.transpose(2, 1)
            arr = preds[:, 0, :].squeeze(0)
            probs = F.softmax(arr, dim=0)
            if self._ema_probs is None:
                self._ema_probs = probs.detach().cpu()
            else:
                self._ema_probs = SMOOTH_WINDOW * self._ema_probs + (1.0 - SMOOTH_WINDOW) * probs.detach().cpu()
        return self._accept_decision(self._ema_probs)

    def init_models(self, weights_path: str, num_classes: int) -> Dict[str, Any]:
        self.i3d = InceptionI3d(400, in_channels=3)
        self.i3d.replace_logits(num_classes)
        if not os.path.exists(weights_path):
            return {"ok": False, "error": f"weights not found: {weights_path}"}
        self.i3d.load_state_dict(torch.load(weights_path, map_location="cuda"))
        self.i3d.cuda()
        self.i3d = nn.DataParallel(self.i3d).eval()

        self.tokenizer = None
        self.k2t_model = None
        try:
            if os.path.isdir("./k2t"):
                self.tokenizer = AutoTokenizer.from_pretrained("./k2t", local_files_only=True)
                self.k2t_model = AutoModelForSeq2SeqLM.from_pretrained("./k2t", local_files_only=True)
                print("Loaded key-to-text from ./k2t")
            else:
                print("No ./k2t directory. Falling back to simple keyword join.")
        except Exception as e:
            print("Warning: failed to load ./k2t:", e)
            print("Fallback to keyword join.")

        self._create_WLASL_dictionary()
        print("TTS: Using Google TTS (gTTS)")
        
        self._ema_probs = None
        self._consec_ok = 0
        self._last_emitted.clear()
        return {"ok": True}

    def start_loop(self, on_update: Callable[[str, np.ndarray], None], batch: int = 40):
        if self._run_flag:
            return
        self._run_flag = True

        def loop():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                on_update("[Error] Cannot open camera", None)
                self._run_flag = False
                return

            frames: List[np.ndarray] = []
            offset = 0
            font = cv2.FONT_HERSHEY_PLAIN
            text_list: List[str] = []
            word_list: List[str] = []
            sentence = ""
            text_count = 0
            last_emit = 0.0

            try:
                while self._run_flag:
                    ret, frame1 = cap.read()
                    if not ret:
                        on_update("[Error] Failed to read frame", None)
                        break

                    h, w, c = frame1.shape
                    sx = 224.0 / w
                    sy = 224.0 / h
                    frame_small = cv2.resize(frame1, dsize=(0, 0), fx=sx, fy=sy)
                    frame_disp = cv2.resize(frame1, (1280, 720))
                    frame_norm = (frame_small / 255.0) * 2.0 - 1.0
                    offset += 1

                    def try_decode():
                        nonlocal sentence, text_count
                        clip = np.asarray(frames, dtype=np.float32)
                        clip_t = torch.from_numpy(clip).permute(3, 0, 1, 2)
                        text = self._run_on_tensor(clip_t)
                        if text != " ":
                            text_count += 1
                            if (text_list and word_list and text_list[-1] != text and word_list[-1] != text) or not text_list:
                                text_list.append(text)
                                word_list.append(text)
                                sentence = f"{sentence} {text}".strip()

                            if text_count >= SENTENCE_TRIGGER:
                                sentence = self._generate_text_from_keywords(text_list)
                                text_count = 0

                        cv2.putText(frame_disp, sentence, (120, 520), font, 0.9, (0, 255, 255), 2, cv2.LINE_4)
                        if len(text_list) > MAX_CONTEXT_WORDS:
                            text_list[:] = text_list[-MAX_CONTEXT_WORDS:]
                            word_list[:] = word_list[-MAX_CONTEXT_WORDS:]

                    if len(frames) < batch:
                        frames.append(frame_norm)
                        if len(frames) == batch:
                            try:
                                try_decode()
                            except Exception:
                                pass
                    else:
                        frames.pop(0)
                        frames.append(frame_norm)
                        if offset % 20 == 0:
                            try:
                                try_decode()
                            except Exception:
                                pass

                    now = time.time()
                    if now - last_emit > 1 / 20:
                        on_update(sentence, frame_disp)
                        last_emit = now
                    time.sleep(0.001)

                cap.release()
            finally:
                cap.release()
                self._run_flag = False

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_loop(self):
        self._run_flag = False
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=1.0)

#UI
import sys
import threading
from PyQt6 import QtWidgets, QtGui, QtCore
import numpy as np

class VideoWidget(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #0f1218; border: 1px solid #2c3340; border-radius: 8px;")

    def update_frame(self, frame_bgr: np.ndarray):
        if frame_bgr is None:
            return
        import cv2
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(pix.scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))

class UpdateSignal(QtCore.QObject):
    updated = QtCore.pyqtSignal(str, object)

class MainWindow(QtWidgets.QWidget):
    def __init__(self, weights_path: str, num_classes: int):
        super().__init__()
        self.setWindowTitle("délApp - Sign Language Interpreter and Companion")
        self.setStyleSheet("""
            QWidget { background-color: #0e0f13; color: #eaeaea; font-family: Segoe UI, Roboto, sans-serif; }
            QPushButton { background-color: #1f6feb; border: none; padding: 10px 16px; border-radius: 6px; color: white; }
            QPushButton:disabled { background-color: #2a2f3a; color: #8a8f98; }
            QPushButton#stop { background-color: #c93c3c; }
            QPushButton#speak { background-color: #10b981; }
            QTextEdit { background-color: #0f1218; border: 1px solid #2c3340; border-radius: 8px; padding: 8px; }
            QLabel#status { color: #9aa4b2; }
            QLabel.section { font-size: 16px; font-weight: 600; color: #cfd7e3; }
            QComboBox { background-color: #1f2937; border: 1px solid #374151; padding: 6px; border-radius: 4px; color: #eaeaea; }
        """)

        self.ctx = TranslatorContext()
        self.weights_path = weights_path
        self.num_classes = num_classes

        self.video = VideoWidget()
        self.video.setMinimumSize(640, 360)

        self.wordsBox = QtWidgets.QTextEdit()
        self.wordsBox.setReadOnly(True)
        self.wordsBox.setPlaceholderText("Recognized words (English) will appear here...")

        self.translatedBox = QtWidgets.QTextEdit()
        self.translatedBox.setReadOnly(True)
        self.translatedBox.setPlaceholderText("Translation will appear here when you click Speak...")

        self.statusLabel = QtWidgets.QLabel("Initializing...")
        self.statusLabel.setObjectName("status")

        self.startBtn = QtWidgets.QPushButton("Start Translating")
        self.stopBtn = QtWidgets.QPushButton("Stop Translating")
        self.stopBtn.setObjectName("stop")
        self.stopBtn.setEnabled(False)

        self.speakBtn = QtWidgets.QPushButton("Speak")
        self.speakBtn.setObjectName("speak")
        self.speakBtn.setEnabled(False)

        self.langLabel = QtWidgets.QLabel("Speech Language:")
        self.langLabel.setProperty("class", "section")
        self.langCombo = QtWidgets.QComboBox()
        self.langCombo.addItems(["English", "Hindi", "Spanish"])
        self.langCombo.currentTextChanged.connect(self.on_language_changed)

        btnRow = QtWidgets.QHBoxLayout()
        btnRow.addWidget(self.startBtn)
        btnRow.addWidget(self.stopBtn)
        btnRow.addWidget(self.speakBtn)
        btnRow.addStretch()

        langRow = QtWidgets.QHBoxLayout()
        langRow.addWidget(self.langLabel)
        langRow.addWidget(self.langCombo)
        langRow.addStretch()

        left = QtWidgets.QVBoxLayout()
        camLabel = QtWidgets.QLabel("Camera")
        camLabel.setProperty("class", "section")
        left.addWidget(camLabel)
        left.addWidget(self.video)
        left.addLayout(btnRow)
        left.addLayout(langRow)
        left.addWidget(self.statusLabel)

        right = QtWidgets.QVBoxLayout()
        txtLabel = QtWidgets.QLabel("Recognized Words (English)")
        txtLabel.setProperty("class", "section")
        right.addWidget(txtLabel)
        right.addWidget(self.wordsBox)
        
        transLabel = QtWidgets.QLabel("Translation")
        transLabel.setProperty("class", "section")
        right.addWidget(transLabel)
        right.addWidget(self.translatedBox)

        root = QtWidgets.QHBoxLayout(self)
        root.addLayout(left, 3)
        root.addLayout(right, 2)

        self.overlay = QtWidgets.QWidget(self)
        self.overlay.setStyleSheet("background-color: rgba(14, 15, 19, 0.85);")
        self.overlay_label = QtWidgets.QLabel("Loading models...")
        self.overlay_label.setStyleSheet("color: #cfd7e3; font-size: 20px;")
        self.overlay_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ovl = QtWidgets.QVBoxLayout(self.overlay)
        ovl.addStretch()
        ovl.addWidget(self.overlay_label)
        spinner = QtWidgets.QProgressBar()
        spinner.setRange(0, 0)
        spinner.setTextVisible(False)
        spinner.setFixedHeight(6)
        spinner.setStyleSheet("QProgressBar { background: #1b2130; border-radius: 3px; } QProgressBar::chunk { background: #1f6feb; }")
        ovl.addWidget(spinner)
        ovl.addStretch()

        self._sig = UpdateSignal()
        self._sig.updated.connect(self._on_update)

        self.startBtn.clicked.connect(self.on_start)
        self.stopBtn.clicked.connect(self.on_stop)
        self.speakBtn.clicked.connect(self.on_speak)

        QtCore.QTimer.singleShot(50, self._kickoff_init)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.overlay.setGeometry(self.rect())

    def _kickoff_init(self):
        self.overlay.show()
        self.statusLabel.setText("Initializing models...")
        t = threading.Thread(target=self._init_thread, daemon=True)
        t.start()

    def _init_thread(self):
        res = self.ctx.init_models(self.weights_path, self.num_classes)
        ok = bool(res.get("ok", False))
        err = res.get("error", "")
        QtCore.QMetaObject.invokeMethod(
            self, "_on_init_done",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(bool, ok),
            QtCore.Q_ARG(str, err)
        )

    @QtCore.pyqtSlot(bool, str)
    def _on_init_done(self, ok: bool, err: str):
        self.overlay.hide()
        if ok:
            self.statusLabel.setText("Models ready. Click Start Translating.")
            self.startBtn.setEnabled(True)
        else:
            self.statusLabel.setText(f"Initialization failed: {err}")
            self.startBtn.setEnabled(False)

    def on_language_changed(self, lang: str):
        self.ctx.set_tts_language(lang)
        # Clear previous translation when language changes
        self.translatedBox.clear()

    def on_start(self):
        self.startBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.speakBtn.setEnabled(False)
        self.translatedBox.clear()
        self.statusLabel.setText("Starting camera...")

        def on_update(words_text: str, frame_bgr: np.ndarray):
            self._sig.updated.emit(words_text or "", frame_bgr)

        self.ctx.start_loop(on_update)
        self.statusLabel.setText("Translating...")

    def on_stop(self):
        self.stopBtn.setEnabled(False)
        self.statusLabel.setText("Stopping...")
        self.ctx.stop_loop()
        self.statusLabel.setText("Stopped.")
        self.startBtn.setEnabled(True)
        text = self.wordsBox.toPlainText().strip()
        self.speakBtn.setEnabled(bool(text))

    def on_speak(self):
        text = self.wordsBox.toPlainText().strip()
        if text:
            # Show translation in the translation box
            translated = self.ctx.translate_text(text, self.ctx.tts_language)
            self.translatedBox.setPlainText(translated)
            # Speak the translated text
            self.ctx.speak(text)

    @QtCore.pyqtSlot(str, object)
    def _on_update(self, words_text: str, frame_bgr: object):
        if words_text:
            if len(words_text) > 1000:
                words_text = words_text[-1000:]
            self.wordsBox.setPlainText(words_text)
            cursor = self.wordsBox.textCursor()
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
            self.wordsBox.setTextCursor(cursor)
        if frame_bgr is not None:
            self.video.update_frame(frame_bgr)

def main():
    weights = "archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt"
    num_classes = 100
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(weights, num_classes)
    w.resize(1200, 720)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
