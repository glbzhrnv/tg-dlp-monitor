import easyocr
import logging
import numpy as np
from PIL import Image
import io
import cv2
import re

logger = logging.getLogger("OCR")

reader = None

def init_easyocr():
    global reader
    if reader is None:
        logger.info("Инициализация EasyOCR (ru + en)...")
        reader = easyocr.Reader(['ru', 'en'], gpu=False)
    return reader

def enhance_image(image_np: np.ndarray) -> np.ndarray:
    try:
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np.copy()
        
        # 1. Удаление шума 
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. Автоматическая коррекция яркости и контраста (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Повышение резкости 
        kernel_sharpen = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        # 4. Адаптивная бинаризация для лучшего контраста
        
        # Метод 1: Адаптивная гауссова
        binary_gauss = cv2.adaptiveThreshold(sharpened, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 15, 8)
        
        # Метод 2: Адаптивная средняя
        binary_mean = cv2.adaptiveThreshold(sharpened, 255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 15, 8)
        
        # Метод 3: Otsu 
        _, binary_otsu = cv2.threshold(sharpened, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        binary = cv2.bitwise_or(binary_gauss, binary_mean)
        binary = cv2.bitwise_or(binary, binary_otsu)
        
        # 5. Морфологическая обработка (удаление мелкого шума)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 6. Увеличение размера для лучшего распознавания мелкого текста
        height, width = binary.shape
        if height < 1000 and width < 1000:
            scale = min(3.0, 2000 / max(height, width))
            new_width = int(width * scale)
            new_height = int(height * scale)
            binary = cv2.resize(binary, (new_width, new_height), 
                              interpolation=cv2.INTER_CUBIC)
        
        return binary
        
    except Exception as e:
        logger.warning(f"Ошибка предобработки: {e}")
        return image_np

def try_multiple_preprocessings(image_np: np.ndarray) -> list:
    results = []
    
    results.append(('original', image_np))
    
    enhanced = enhance_image(image_np)
    results.append(('enhanced', enhanced))
    
    # Инвертированные цвета 
    if len(image_np.shape) == 2:
        inverted = cv2.bitwise_not(image_np)
        results.append(('inverted', inverted))
    
    # Увеличение контраста для плохих фото
    if len(image_np.shape) == 2:
        high_contrast = cv2.equalizeHist(image_np)
        results.append(('high_contrast', high_contrast))
    
    # Удаление фона
    try:
        if len(image_np.shape) == 2:
            # Используем медианный фильтр для оценки фона
            background = cv2.medianBlur(image_np, 51)
            foreground = cv2.absdiff(image_np, background)
            _, bg_removed = cv2.threshold(foreground, 30, 255, cv2.THRESH_BINARY)
            results.append(('bg_removed', bg_removed))
    except:
        pass
    
    return results

def correct_ocr_text(text: str) -> str:
    if not text:
        return text
    
    # Типичные ошибки распознавания для кириллицы
    corrections = {
        '0': 'О', '0': 'о',  # Ноль как буква О
        '3': 'З', '3': 'з',
        '4': 'Ч', '4': 'ч',
        '6': 'б',  # 6 как б
        '8': 'В', '8': 'в',
        '1': 'l', '1': 'I',
        '5': 'S',
        '7': 'T',
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u0400-\u04FF\-\.\,]', '', text)
    
    return text.strip()

def extract_text_from_image(image_bytes: bytes) -> str:
    if not image_bytes or len(image_bytes) < 100:
        return ""
    
    best_text = ""
    best_confidence = 0
    
    try:
        reader = init_easyocr()
        
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_np = np.array(image)
        
        preprocessing_variants = try_multiple_preprocessings(image_np)
        
        # Варианты параметров EasyOCR
        param_variants = [
            {'detail': 0, 'paragraph': True, 'min_size': 5, 'text_threshold': 0.5},
            {'detail': 0, 'paragraph': False, 'min_size': 3, 'text_threshold': 0.3},
            {'detail': 0, 'paragraph': True, 'min_size': 5, 'text_threshold': 0.7, 'contrast_ths': 0.1},
            {'detail': 0, 'paragraph': False, 'min_size': 5, 'text_threshold': 0.4, 'width_ths': 0.7},
        ]
        
        # Перебираем комбинации предобработки и параметров
        for prep_name, prep_image in preprocessing_variants:
            if prep_image.shape[0] < 50 or prep_image.shape[1] < 50:
                continue
                
            for params in param_variants:
                try:
                    result = reader.readtext(prep_image, **params)
                    
                    if result:
                        text = " ".join(result).strip()
                        
                        text = ''.join(c for c in text if c.isprintable() or c.isspace())
                        text = ' '.join(text.split()).strip()
                        
                        text = correct_ocr_text(text)
                        
                        # Оцениваем качество
                        if text:
                            good_chars = sum(1 for c in text if c.isalnum())
                            total_chars = len(text)
                            
                            if total_chars > 0:
                                quality = good_chars / total_chars
                                
                                confidence = quality * min(1.0, len(text) / 100)
                                
                                cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
                                if cyrillic_count > 0:
                                    confidence += 0.1
                                
                                if confidence > best_confidence:
                                    best_text = text
                                    best_confidence = confidence
                                    logger.info(f"Найден хороший текст ({prep_name}, params={params.get('text_threshold')}): {text[:100]}...")
                    
                except Exception as e:
                    logger.debug(f"Ошибка OCR ({prep_name}, {params}): {e}")
                    continue
        
        if not best_text:
            logger.info("Пробуем базовый OCR без предобработки")
            basic_result = reader.readtext(image_np, detail=0, paragraph=True)
            if basic_result:
                best_text = " ".join(basic_result).strip()
                best_text = correct_ocr_text(best_text)
        
        if best_text:
            logger.info(f"OCR успешно извлёк {len(best_text)} символов: {best_text[:150]}...")
        else:
            logger.info("OCR ничего не нашёл на изображении")
        
        return best_text if best_text else "[текст не распознан]"
        
    except Exception as e:
        logger.error(f"Ошибка OCR: {e}", exc_info=True)
        return f"[ошибка OCR: {str(e)}]"