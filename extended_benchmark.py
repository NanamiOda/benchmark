"""
Benchmark extendido para Speech-to-Text con modelos open source ligeros
Incluye análisis de eficiencia y alternativas ligeras
"""

import os
import time
import json
import psutil
import threading
import whisper
import torch
import numpy as np
import difflib
import gc  # Para garbage collection
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Para métricas WER y CER
import jiwer
from jiwer import wer, cer

# Modelos adicionales open source
try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class ExtendedPerformanceMonitor:
    """Monitor extendido de rendimiento con métricas adicionales"""
    
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.monitoring = False
        self.start_time = None
        self.end_time = None
        self.start_memory_mb = None
        self.peak_memory_mb = None
    
    def start_monitoring(self):
        self.monitoring = True
        self.start_time = time.time()
        self.cpu_usage = []
        self.memory_usage = []
        
        # Memoria inicial en MB
        process = psutil.Process()
        self.start_memory_mb = process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = self.start_memory_mb
        
        def monitor():
            while self.monitoring:
                self.cpu_usage.append(psutil.cpu_percent())
                self.memory_usage.append(psutil.virtual_memory().percent)
                
                # Actualizar memoria pico del proceso
                current_memory = process.memory_info().rss / 1024 / 1024
                if current_memory > self.peak_memory_mb:
                    self.peak_memory_mb = current_memory
                
                time.sleep(0.1)
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        self.end_time = time.time()
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def get_metrics(self):
        # Manejar caso donde monitoring nunca se inició
        if self.start_time is None or self.end_time is None:
            return {
                'elapsed_time_ms': 0,
                'avg_cpu_percent': 0,
                'max_cpu_percent': 0,
                'avg_memory_percent': 0,
                'max_memory_percent': 0,
                'memory_used_mb': 0,
                'peak_memory_mb': 0
            }
        
        elapsed_time = (self.end_time - self.start_time) * 1000  # en ms
        avg_cpu = np.mean(self.cpu_usage) if self.cpu_usage else 0
        max_cpu = np.max(self.cpu_usage) if self.cpu_usage else 0
        avg_memory = np.mean(self.memory_usage) if self.memory_usage else 0
        max_memory = np.max(self.memory_usage) if self.memory_usage else 0
        
        memory_used = self.peak_memory_mb - self.start_memory_mb if self.start_memory_mb else 0
        
        return {
            'elapsed_time_ms': elapsed_time,
            'avg_cpu_percent': avg_cpu,
            'max_cpu_percent': max_cpu,
            'avg_memory_percent': avg_memory,
            'max_memory_percent': max_memory,
            'memory_used_mb': memory_used,
            'peak_memory_mb': self.peak_memory_mb if self.peak_memory_mb else 0
        }

def limpiar_memoria():
    """Forzar limpieza de memoria y caché de modelos"""
    try:
        # Limpiar caché de PyTorch si está disponible
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Forzar garbage collection múltiple
        for _ in range(3):
            gc.collect()
        
        print("    Memoria limpiada")
        
    except Exception as e:
        print(f"    Warning limpiando memoria: {e}")

class ExtendedSpeechBenchmark:
    """Benchmark extendido para múltiples modelos open source"""
    
    def __init__(self, audio_dir="audioCuentos", reference_dir="textoReal"):
        self.audio_dir = Path(audio_dir)
        self.reference_dir = Path(reference_dir)
        self.results = []
        
        # Archivos de test
        self.test_files = {
            "caperucita.mp3": "caperucitaReal.txt",
            "los3cerditos.mp3": "los3cerditosReal.txt", 
            "pinocho.mp3": "pinochoReal.txt"
        }
    
    def load_reference_text(self, ref_file):
        """Cargar texto de referencia"""
        try:
            with open(self.reference_dir / ref_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except UnicodeDecodeError:
            with open(self.reference_dir / ref_file, 'r', encoding='latin1') as f:
                return f.read().strip()
    
    def normalize_text(self, text):

        import re
        import unicodedata
        
        if not text:
            return ""
        
        text = text.lower()
        
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        
        text = re.sub(r'[¡¿"""''""„‚‛‟]', '', text)  # Signos de apertura/cierre
        text = re.sub(r'[.!?:;,\-—\(\)\[\]{}]', ' ', text)  # Puntuación básica
        
        text = re.sub(r'\s+', ' ', text)
        
        text = text.strip()
        
        return text
    
    def calculate_metrics(self, reference, hypothesis):
        """Calcular WER y CER"""
        try:
            # Normalizar ambos textos
            ref_clean = self.normalize_text(reference)
            hyp_clean = self.normalize_text(hypothesis)
            
            print(f"    REF (primeros 150 chars): '{ref_clean[:150]}...'")
            print(f"    HYP (primeros 150 chars): '{hyp_clean[:150]}...'")
            
            # Validar que tenemos texto para comparar
            if not ref_clean.strip():
                print("    Texto de referencia vacío después de normalización")
                return {
                    'wer': 1.0, 'cer': 1.0, 'word_count_diff': 1.0, 'char_count_diff': 1.0,
                    'ref_words': 0, 'hyp_words': 0, 'ref_chars': 0, 'hyp_chars': 0,
                    'ref_normalized': ref_clean, 'hyp_normalized': hyp_clean
                }
            
            if not hyp_clean.strip():
                print("    Transcripción vacía después de normalización")
                return {
                    'wer': 1.0, 'cer': 1.0, 'word_count_diff': 1.0, 'char_count_diff': 1.0,
                    'ref_words': len(ref_clean.split()), 'hyp_words': 0, 
                    'ref_chars': len(ref_clean), 'hyp_chars': 0,
                    'ref_normalized': ref_clean, 'hyp_normalized': hyp_clean
                }
            
            try:
                # Para WER y CER, usar las cadenas directamente (jiwer maneja la tokenización)
                wer_score = jiwer.wer(ref_clean, hyp_clean)
                cer_score = jiwer.cer(ref_clean, hyp_clean)
                    
            except Exception as metric_error:
                print(f"    Error en cálculo jiwer: {metric_error}")
                # Fallback a cálculo manual
                ref_words_list = ref_clean.split()
                hyp_words_list = hyp_clean.split()
                
                if not ref_words_list:
                    wer_score = 1.0 if hyp_words_list else 0.0
                else:
                    # WER manual básico usando difflib
                    matcher = difflib.SequenceMatcher(None, ref_words_list, hyp_words_list)
                    wer_score = 1.0 - matcher.ratio()
                
                if not ref_clean:
                    cer_score = 1.0 if hyp_clean else 0.0
                else:
                    # CER manual básico usando difflib
                    char_matcher = difflib.SequenceMatcher(None, ref_clean.replace(' ', ''), hyp_clean.replace(' ', ''))
                    cer_score = 1.0 - char_matcher.ratio()
            
            # Métricas adicionales
            ref_words = len(ref_clean.split())
            hyp_words = len(hyp_clean.split())
            ref_chars = len(ref_clean.replace(' ', ''))  # Caracteres sin espacios
            hyp_chars = len(hyp_clean.replace(' ', ''))
            
            word_diff = abs(ref_words - hyp_words) / ref_words if ref_words > 0 else 0
            char_diff = abs(ref_chars - hyp_chars) / ref_chars if ref_chars > 0 else 0
            
            # Información de debug
            print(f"    Ref: {ref_words} palabras, {ref_chars} chars")
            print(f"    Hyp: {hyp_words} palabras, {hyp_chars} chars")
            print(f"    WER calculado: {wer_score*100:.1f}%, CER calculado: {cer_score*100:.1f}%")
            
            return {
                'wer': wer_score,
                'cer': cer_score,
                'word_count_diff': word_diff,
                'char_count_diff': char_diff,
                'ref_words': ref_words,
                'hyp_words': hyp_words,
                'ref_chars': ref_chars,
                'hyp_chars': hyp_chars,
                'ref_normalized': ref_clean,
                'hyp_normalized': hyp_clean
            }
            
        except Exception as e:
            print(f"    Error calculando métricas: {e}")
            import traceback
            traceback.print_exc()
            return {
                'wer': 1.0, 'cer': 1.0, 'word_count_diff': 1.0, 'char_count_diff': 1.0,
                'ref_words': 0, 'hyp_words': 0, 'ref_chars': 0, 'hyp_chars': 0,
                'ref_normalized': '', 'hyp_normalized': ''
            }
    
    def test_whisper_model(self, model_size, audio_file):
        """Test genérico para cualquier modelo Whisper"""
        monitor = ExtendedPerformanceMonitor()
        
        # Limpiar memoria antes de cargar el modelo
        print(f"    Limpiando memoria antes de cargar {model_size}...")
        limpiar_memoria()
        
        try:
            # Usar archivos locales si están disponibles
            local_model_files = {
                "tiny": "tiny.pt",
                "base": "base.pt"
            }
            
            if model_size in local_model_files:
                model_path = local_model_files[model_size]
                if os.path.exists(model_path):
                    print(f"    Cargando modelo local {model_path}...")
                    model = whisper.load_model(model_path)
                else:
                    print(f"    Modelo local {model_path} no encontrado, descargando {model_size}...")
                    model = whisper.load_model(model_size)
            else:
                print(f"    Cargando Whisper {model_size}...")
                model = whisper.load_model(model_size)
            
            audio_path = str(self.audio_dir / audio_file)
            
            print(f"    Transcribiendo...")
            monitor.start_monitoring()
            
            result = model.transcribe(
                audio_path, 
                language="es",
                fp16=torch.cuda.is_available()
            )
            
            monitor.stop_monitoring()
            
            transcription = result["text"].strip()
            metrics = monitor.get_metrics()
            
            # Información adicional del modelo
            model_info = {
                'model_size_name': model_size,
                'device': 'GPU' if torch.cuda.is_available() else 'CPU',
                'fp16': torch.cuda.is_available(),
                'local_model': model_size in local_model_files and os.path.exists(local_model_files[model_size])
            }
            
            # Limpiar memoria después del test
            del model  # Eliminar referencia explícitamente
            limpiar_memoria()
            
            return transcription, {**metrics, **model_info}, True
            
        except Exception as e:
            monitor.stop_monitoring()
            # Limpiar memoria incluso en caso de error
            limpiar_memoria()
            error_msg = str(e)
            if "[WinError 2]" in error_msg or "FileNotFoundError" in str(type(e)):
                print(f"    Error: FFmpeg no encontrado. Instálalo: https://ffmpeg.org/download.html")
            else:
                print(f"    Error en Whisper {model_size}: {e}")
            return "", monitor.get_metrics(), False
    
    def test_vosk_model(self, audio_file):
        """Test con Vosk usando modelo automático"""
        if not VOSK_AVAILABLE:
            print("    Vosk no disponible")
            return "", {}, False
        
        monitor = ExtendedPerformanceMonitor()
        
        try:
            import wave
            import json
            import subprocess
            
            print("    Cargando modelo Vosk (español)...")
            
            # Limpiar memoria antes de cargar el modelo
            limpiar_memoria()
            
            # Usar modelo automático de Vosk
            model = vosk.Model(lang="es")
            
            # Convertir audio
            audio_path = str(self.audio_dir / audio_file)
            wav_path = audio_path.replace('.mp3', '_temp_vosk.wav')
            
            print("    Convirtiendo audio...")
            subprocess.run([
                'ffmpeg', '-i', audio_path, '-ar', '16000', '-ac', '1', 
                wav_path, '-y'
            ], capture_output=True, check=True)
            
            print("    Transcribiendo...")
            monitor.start_monitoring()
            
            wf = wave.open(wav_path, 'rb')
            rec = vosk.KaldiRecognizer(model, wf.getframerate())
            
            transcription_parts = []
            
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if 'text' in result and result['text'].strip():
                        transcription_parts.append(result['text'])
                else:
                    # También capturar resultados parciales si es necesario
                    partial = json.loads(rec.PartialResult())
                    if 'partial' in partial and partial['partial'].strip():
                        pass  # Los parciales se procesan en el siguiente AcceptWaveform
            
            # Obtener resultado final
            final_result = json.loads(rec.FinalResult())
            if 'text' in final_result and final_result['text'].strip():
                transcription_parts.append(final_result['text'])
            
            monitor.stop_monitoring()
            wf.close()
            
            # Limpiar archivo temporal
            os.remove(wav_path)
            
            # Unir todas las partes de la transcripción
            transcription = " ".join(transcription_parts).strip()
            
            model_info = {
                'model_size_name': 'vosk-auto',
                'device': 'CPU',
                'fp16': False
            }
            
            # Limpiar memoria después del test
            del model, rec
            limpiar_memoria()
            
            return transcription, {**monitor.get_metrics(), **model_info}, True
            
        except Exception as e:
            monitor.stop_monitoring()
            # Limpiar memoria incluso en caso de error
            limpiar_memoria()
            error_msg = str(e)
            if "[WinError 2]" in error_msg or "FileNotFoundError" in str(type(e)):
                print(f"    Error: FFmpeg no encontrado. Instálalo: https://ffmpeg.org/download.html")
            else:
                print(f"    Error en Vosk: {e}")
            # Limpiar archivo temporal si existe
            wav_path = str(self.audio_dir / audio_file).replace('.mp3', '_temp_vosk.wav')
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return "", monitor.get_metrics(), False
    
    def test_speech_recognition_google(self, audio_file):
        """Test con SpeechRecognition + Google API (requiere internet)"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            print("    SpeechRecognition no disponible")
            return "", {}, False
        
        monitor = ExtendedPerformanceMonitor()
        
        try:
            import subprocess
            
            r = sr.Recognizer()
            audio_path = str(self.audio_dir / audio_file)
            wav_path = audio_path.replace('.mp3', '_temp_sr.wav')
            
            # Convertir a WAV
            subprocess.run([
                'ffmpeg', '-i', audio_path, '-ar', '16000', '-ac', '1', 
                wav_path, '-y'
            ], capture_output=True, check=True)
            
            print("    Transcribiendo con Google API...")
            monitor.start_monitoring()
            
            with sr.AudioFile(wav_path) as source:
                audio = r.record(source)
                transcription = r.recognize_google(audio, language='es-ES')
            
            monitor.stop_monitoring()
            
            # Limpiar
            os.remove(wav_path)
            
            model_info = {
                'model_size_name': 'google-api',
                'device': 'Cloud',
                'fp16': False
            }
            
            return transcription, {**monitor.get_metrics(), **model_info}, True
            
        except Exception as e:
            monitor.stop_monitoring()
            error_msg = str(e)
            if "[WinError 2]" in error_msg or "FileNotFoundError" in str(type(e)):
                print(f"    Error: FFmpeg no encontrado. Instálalo: https://ffmpeg.org/download.html")
            else:
                print(f"    Error en Google API: {e}")
            return "", monitor.get_metrics(), False
    

    
    def run_extended_benchmark(self):
        """Ejecutar benchmark extendido"""
        print("BENCHMARK EXTENDIDO SPEECH-TO-TEXT")
        print("=" * 60)
        
        # Modelos a testear - tiny, small y base
        models_to_test = [
            ("Whisper Tiny", lambda af: self.test_whisper_model("tiny", af)),
            ("Whisper Small", lambda af: self.test_whisper_model("small", af)),
            ("Whisper Base", lambda af: self.test_whisper_model("base", af)),
        ]
        
        if VOSK_AVAILABLE:
            models_to_test.append(("Vosk Small", self.test_vosk_model))
        
        if SPEECH_RECOGNITION_AVAILABLE:
            models_to_test.append(("Google API", self.test_speech_recognition_google))
        
        print(f"Modelos a evaluar: {len(models_to_test)}")
        print(f"Archivos de audio: {len(self.test_files)}")
        print(f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
        print()
        
        total_tests = len(self.test_files) * len(models_to_test)
        current_test = 0
        
        for audio_file, ref_file in self.test_files.items():
            print(f"\nProcesando: {audio_file}")
            print("-" * 50)
            
            # Cargar referencia
            reference_text = self.load_reference_text(ref_file)
            print(f"Referencia: {len(reference_text)} chars, {len(reference_text.split())} words")
            
            for model_name, test_function in models_to_test:
                current_test += 1
                print(f"\n  [{current_test}/{total_tests}] {model_name}")
                
                # Ejecutar test
                transcription, performance_metrics, success = test_function(audio_file)
                
                if success:
                    # Calcular métricas de precisión
                    accuracy_metrics = self.calculate_metrics(reference_text, transcription)
                    
                    result = {
                        'audio_file': audio_file,
                        'model': model_name,
                        'transcription': transcription,
                        'reference': reference_text,
                        'success': True,
                        **accuracy_metrics,
                        **performance_metrics
                    }
                    
                    print(f"    WER: {accuracy_metrics['wer']*100:.1f}% | CER: {accuracy_metrics['cer']*100:.1f}%")
                    print(f"    Tiempo: {performance_metrics['elapsed_time_ms']:.0f}ms")
                    print(f"    Memoria: {performance_metrics.get('memory_used_mb', 0):.1f}MB")
                    
                else:
                    result = {
                        'audio_file': audio_file,
                        'model': model_name,
                        'transcription': '',
                        'reference': reference_text,
                        'success': False,
                        'wer': 1.0, 'cer': 1.0,
                        **performance_metrics
                    }
                    print(f"    Falló")
                
                self.results.append(result)
        
        print(f"\nBenchmark completado! ({len(self.results)} tests)")
        return self.results
    
    def generate_comprehensive_report(self):
        """Generar reporte comprehensivo"""
        if not self.results:
            print("No hay resultados")
            return
        
        df = pd.DataFrame(self.results)
        successful = df[df['success'] == True]
        
        if successful.empty:
            print("No hay resultados exitosos")
            return
        
        print("\nREPORTE COMPREHENSIVO")
        print("=" * 60)
        
        # Ranking por precisión
        print("\nRANKING POR PRECISIÓN (WER):")
        print("-" * 40)
        wer_ranking = successful.groupby('model')['wer'].mean().sort_values()
        for i, (model, wer_val) in enumerate(wer_ranking.items(), 1):
            print(f"{i}. {model}: {wer_val:.3f}")
        
        # Ranking por velocidad
        print("\nRANKING POR VELOCIDAD:")
        print("-" * 40)
        speed_ranking = successful.groupby('model')['elapsed_time_ms'].mean().sort_values()
        for i, (model, time_val) in enumerate(speed_ranking.items(), 1):
            print(f"{i}. {model}: {time_val:.0f}ms")
        
        # Ranking por eficiencia de memoria
        print("\nRANKING POR EFICIENCIA DE MEMORIA:")
        print("-" * 40)
        memory_ranking = successful.groupby('model')['memory_used_mb'].mean().sort_values()
        for i, (model, mem_val) in enumerate(memory_ranking.items(), 1):
            print(f"{i}. {model}: {mem_val:.1f}MB")
        
        # Análisis de trade-offs
        print("\nANÁLISIS DE TRADE-OFFS:")
        print("-" * 40)
        for model in successful['model'].unique():
            model_data = successful[successful['model'] == model]
            avg_wer = model_data['wer'].mean()
            avg_time = model_data['elapsed_time_ms'].mean()
            avg_memory = model_data['memory_used_mb'].mean()
            
            # Score combinado (menor es mejor)
            combined_score = (avg_wer * 0.5) + (avg_time / 10000 * 0.3) + (avg_memory / 1000 * 0.2)
            
            print(f"{model}:")
            print(f"  Score combinado: {combined_score:.3f}")
            print(f"  WER: {avg_wer*100:.1f}% | Tiempo: {avg_time:.0f}ms | Memoria: {avg_memory:.1f}MB")
        
        # Recomendaciones
        print("\nRECOMENDACIONES:")
        print("-" * 40)
        
        best_accuracy = wer_ranking.index[0]
        best_speed = speed_ranking.index[0]
        best_memory = memory_ranking.index[0]
        
        print(f"Mejor precisión: {best_accuracy}")
        print(f"Más rápido: {best_speed}")
        print(f"Menos memoria: {best_memory}")
        
        if best_accuracy == best_speed == best_memory:
            print(f"GANADOR ABSOLUTO: {best_accuracy}")
        else:
            print("Diferentes modelos destacan en diferentes aspectos")
    
    def create_detailed_visualizations(self):
        """Crear visualizaciones detalladas como gráficos separados"""
        if not self.results:
            print("No hay resultados para graficar")
            return
        
        df = pd.DataFrame(self.results)
        successful = df[df['success'] == True]
        
        if successful.empty:
            print("No hay resultados exitosos para graficar")
            return
        
        # Configurar el estilo
        plt.style.use('default')
        # Paleta de colores diferenciados (verde, naranja, morado)
        color_palette = ['#2ecc71', '#e67e22', '#9b59b6']
        sns.set_palette(color_palette)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Gráfico 1: WER por modelo y archivo
        plt.figure(figsize=(12, 8))
        wer_order = successful.groupby('model')['wer'].mean().sort_values(ascending=False).index
        ax1 = sns.barplot(data=successful, y='model', x='wer', hue='audio_file', order=wer_order, orient='h')
        plt.title('Word Error Rate (WER) por Modelo', fontsize=16, fontweight='bold')
        plt.xlabel('WER Score (menor es mejor)', fontsize=12)
        plt.ylabel('Modelo', fontsize=12)
        plt.legend(title='Archivo', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Agregar valores con porcentaje dentro de las barras
        for container in ax1.containers:
            for bar in container:
                width = bar.get_width()
                percentage = width * 100
                label = f'{width:.3f} ({percentage:.1f}%)'
                ax1.text(width * 0.5, bar.get_y() + bar.get_height()/2., label,
                        ha='center', va='center', fontsize=9, color='white', weight='bold')
        
        plt.tight_layout()
        filename_wer = f"{timestamp}_01_WER.png"
        plt.savefig(filename_wer, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico WER guardado: {filename_wer}")
        
        # Gráfico 2: CER por modelo y archivo
        plt.figure(figsize=(12, 8))
        cer_order = successful.groupby('model')['cer'].mean().sort_values(ascending=False).index
        ax2 = sns.barplot(data=successful, y='model', x='cer', hue='audio_file', order=cer_order, orient='h')
        plt.title('Character Error Rate (CER) por Modelo', fontsize=16, fontweight='bold')
        plt.xlabel('CER Score (menor es mejor)', fontsize=12)
        plt.ylabel('Modelo', fontsize=12)
        plt.legend(title='Archivo', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Agregar valores con porcentaje dentro de las barras
        for container in ax2.containers:
            for bar in container:
                width = bar.get_width()
                percentage = width * 100
                label = f'{width:.3f} ({percentage:.1f}%)'
                ax2.text(width * 0.5, bar.get_y() + bar.get_height()/2., label,
                        ha='center', va='center', fontsize=9, color='white', weight='bold')
        
        plt.tight_layout()
        filename_cer = f"{timestamp}_02_CER.png"
        plt.savefig(filename_cer, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico CER guardado: {filename_cer}")
        
        # Gráfico 3: Tiempo de procesamiento
        plt.figure(figsize=(12, 8))
        # Convertir milisegundos a segundos
        successful['elapsed_time_s'] = successful['elapsed_time_ms'] / 1000

        time_order = successful.groupby('model')['elapsed_time_s'].mean().sort_values(ascending=True).index
        ax3 = sns.barplot(data=successful, y='model', x='elapsed_time_s', hue='audio_file', order=time_order, orient='h')
        plt.title('Tiempo de Procesamiento', fontsize=16, fontweight='bold')
        plt.xlabel('Tiempo (segundos - menor es mejor)', fontsize=12)
        plt.ylabel('Modelo', fontsize=12)
        plt.legend(title='Archivo', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Agregar solo valores en segundos dentro de las barras
        for container in ax3.containers:
            for bar in container:
                width = bar.get_width()
                label = f'{width:.1f} s'
                ax3.text(width * 0.5, bar.get_y() + bar.get_height()/2., label,
                        ha='center', va='center', fontsize=10, color='white', weight='bold')
        
        plt.tight_layout()
        filename_time = f"{timestamp}_03_Tiempo.png"
        plt.savefig(filename_time, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico Tiempo guardado: {filename_time}")
        
        # Gráfico 4: Scatter plot WER vs Tiempo
        plt.figure(figsize=(12, 8))
        for model in successful['model'].unique():
            model_data = successful[successful['model'] == model]
            plt.scatter(model_data['elapsed_time_ms'], model_data['wer'], 
                       label=model, s=100, alpha=0.7)
        plt.xlabel('Tiempo (ms)', fontsize=12)
        plt.ylabel('WER Score', fontsize=12)
        plt.title('Trade-off: Precisión vs Velocidad', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename_scatter = f"{timestamp}_04_TradeOff.png"
        plt.savefig(filename_scatter, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico Trade-off guardado: {filename_scatter}")
        
        # Gráfico 5: Uso de memoria
        if 'memory_used_mb' in successful.columns:
            plt.figure(figsize=(12, 8))
            memory_order = successful.groupby('model')['memory_used_mb'].mean().sort_values(ascending=False).index
            ax5 = sns.barplot(data=successful, y='model', x='memory_used_mb', order=memory_order, orient='h')
            plt.title('Uso de Memoria', fontsize=16, fontweight='bold')
            plt.xlabel('Memoria (MB - menor es mejor)', fontsize=12)
            plt.ylabel('Modelo', fontsize=12)
            for container in ax5.containers:
                ax5.bar_label(container, fmt='%.1f', padding=3, fontsize=10)
            plt.tight_layout()
            filename_memory = f"{timestamp}_05_Memoria.png"
            plt.savefig(filename_memory, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Gráfico Memoria guardado: {filename_memory}")
        
        # Gráfico 6: Score de eficiencia
        efficiency_data = []
        for model in successful['model'].unique():
            model_data = successful[successful['model'] == model]
            avg_wer = model_data['wer'].mean()
            avg_time = model_data['elapsed_time_ms'].mean()
            avg_memory = model_data.get('memory_used_mb', pd.Series([0])).mean()
            
            norm_wer = avg_wer
            norm_time = avg_time / successful['elapsed_time_ms'].max()
            norm_memory = avg_memory / successful.get('memory_used_mb', pd.Series([1])).max() if avg_memory > 0 else 0
            
            efficiency_score = norm_wer * 0.5 + norm_time * 0.3 + norm_memory * 0.2
            
            efficiency_data.append({
                'model': model,
                'efficiency_score': efficiency_score,
                'avg_wer': avg_wer,
                'avg_time_ms': avg_time
            })
        
        efficiency_df = pd.DataFrame(efficiency_data)
        efficiency_df_sorted = efficiency_df.sort_values('efficiency_score', ascending=False)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(efficiency_df_sorted['model'], efficiency_df_sorted['efficiency_score'])
        plt.title('Score de Eficiencia Combinado', fontsize=16, fontweight='bold')
        plt.xlabel('Score (menor es mejor)', fontsize=12)
        plt.ylabel('Modelo', fontsize=12)
        for bar, score in zip(bars, efficiency_df_sorted['efficiency_score']):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{score:.3f}', ha='left', va='center', fontsize=10)
        plt.tight_layout()
        filename_efficiency = f"{timestamp}_06_Eficiencia.png"
        plt.savefig(filename_efficiency, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico Eficiencia guardado: {filename_efficiency}")
        
        print(f"\nTodos los gráficos han sido guardados con prefijo: {timestamp}")
        
        return efficiency_df
    
    def save_extended_results(self):
        """Guardar resultados extendidos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_extended_benchmark_results.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"Resultados guardados en {filename}")
        return filename

def main():
    print("BENCHMARK EXTENDIDO SPEECH-TO-TEXT")
    print("=" * 60)
    print("Incluye modelos open source y métricas avanzadas")
    print()
    
    # Verificar directorios
    audio_dir = Path("audioCuentos")
    ref_dir = Path("textoReal")
    
    if not audio_dir.exists() or not ref_dir.exists():
        print("Directorios no encontrados")
        return
    
    # Ejecutar benchmark
    benchmark = ExtendedSpeechBenchmark()
    
    try:
        results = benchmark.run_extended_benchmark()
        benchmark.save_extended_results()
        benchmark.generate_comprehensive_report()
        
        # Crear visualizaciones detalladas
        print("\nGenerando gráficos detallados...")
        efficiency_df = benchmark.create_detailed_visualizations()
        
        if efficiency_df is not None and not efficiency_df.empty:
            print("\nRANKING DE EFICIENCIA:")
            print("-" * 30)
            efficiency_sorted = efficiency_df.sort_values('efficiency_score')
            for i, row in efficiency_sorted.iterrows():
                print(f"{efficiency_sorted.index.get_loc(i)+1}. {row['model']}: {row['efficiency_score']:.3f}")
                print(f"   WER: {row['avg_wer']*100:.1f}% | Tiempo: {row['avg_time_ms']:.0f}ms")
        
        print("\n¡Benchmark extendido completado!")
        
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
