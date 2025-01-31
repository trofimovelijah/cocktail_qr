import torch
import torchaudio
import os
import random

class SpeechGenerator:
    def __init__(self):
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v4_ru'
        )
        
        self.style_config = {
            "1": {"speaker": "aidar", "emotion": "sad"},     # Космический ужас
            "2": {"speaker": "eugene", "emotion": "angry"},   # Гопнический стиль
            "3": {"speaker": "baya", "emotion": "surprise"},  # Экспериментальный
            "4": {"speaker": "kseniya", "emotion": "neutral"} # По умолчанию
        }

    async def generate_speech(self, text: str, style: str) -> str:
        try:
            config = self.style_config[style]
            
            # Генерация аудио
            audio = self.model.apply_tts(
                text=text,
                speaker=config["speaker"],
                sample_rate=48000
            )
            
            # Сохранение файла
            output_dir = 'audio'
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f'output_{random.randint(1, 9999)}.wav')
            
            torchaudio.save(
                output_file,
                audio.unsqueeze(0),
                48000
            )
            
            return output_file
            
        except Exception as e:
            print(f"Ошибка генерации речи: {e}")
            return None 