#!/usr/bin/env python3
# ================================================================
# CropAI // ONE AI // 5-CROP UNO Q EDITION
# Unified camera + image upload + crop-specific Keras models
#
# Models supplied for this build:
#   corn      -> corn_disease_best.keras
#   cotton    -> cotton_disease_v8_best.keras
#   paddy     -> paddy_disease_v2_best.keras
#   sugarcane -> sugarcane_disease_v2_best.keras
#   wheat     -> wheat_disease_best.keras
#
# Features:
#   1. Startup profile: name, place, language
#   2. English default + Indian language UI packs
#   3. Five-crop selection screen with generated crop cards
#   4. External webcam
#   5. Capture / recapture
#   6. Image upload
#   7. Crop-specific Keras model selection
#   8. Top-3 prediction + confidence + uncertainty warning
#   9. DHT11 temperature/humidity from UNO Q serial
#  10. Disease cause / management / chemical information
#  11. Save PNG/JPEG
#  12. Save a PDF report containing the captured image and result
#
# IMPORTANT:
#   The model class order MUST match the order used while training.
#   Exact orders for the supplied Paddy/Cotton/Corn models are included.
#   Sugarcane/Wheat class order is kept in one clearly marked section
#   because the uploaded .keras files do not contain folder-name metadata.
#
# DHT11 communication uses Arduino_RouterBridge on the UNO Q MCU.
# The MCU exposes get_dht_temperature() and get_dht_humidity();
# Linux reads them through /var/run/arduino-router.sock using msgpack.
#
# Install on the UNO Q Linux environment as needed:
#   python3 -m pip install tensorflow numpy opencv-python pillow reportlab msgpack deep-translator
#
# ================================================================

import os
import sys
import time
import json
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from urllib.request import Request, urlopen, urlretrieve
from urllib.parse import urljoin
import re
import socket
import itertools

try:
    import msgpack
except Exception:
    msgpack = None

import cv2
import numpy as np

try:
    import serial
except Exception:
    serial = None

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext
except Exception:
    tk = None
    filedialog = None
    messagebox = None

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageTk = None
    ImageDraw = None
    ImageFont = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
except Exception:
    A4 = None
    ImageReader = None
    canvas = None

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None


# ================================================================
# PATHS
# ================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WINDOWS_ROOT = Path(r"G:\CropAI")
LINUX_ROOT = Path("/home/arduino/CropAI")

ROOT_CANDIDATES = []
if os.environ.get("CROP_AI_ROOT"):
    ROOT_CANDIDATES.append(Path(os.environ["CROP_AI_ROOT"]).expanduser())
ROOT_CANDIDATES.extend([REPO_ROOT, WINDOWS_ROOT, LINUX_ROOT])

_seen_roots = set()
_unique_roots = []
for candidate in ROOT_CANDIDATES:
    key = str(candidate)
    if key not in _seen_roots:
        _seen_roots.add(key)
        _unique_roots.append(candidate)
ROOT_CANDIDATES = _unique_roots

MODEL_FILENAMES = {
    "corn": "corn_disease_best.keras",
    "cotton": "cotton_disease_v8_best.keras",
    "paddy": "paddy_disease_v2_best.keras",
    "sugarcane": "sugarcane_disease_v2_best.keras",
    "wheat": "wheat_disease_best.keras",
}

# Repository layout: trained models/keras models/
# UNO Q deployment layout: models/
MODEL_DIR = None
ROOT = REPO_ROOT
for candidate_root in ROOT_CANDIDATES:
    for candidate_dir in (
        candidate_root / "trained models" / "keras models",
        candidate_root / "models",
    ):
        if candidate_dir.exists() and any(
            (candidate_dir / filename).exists()
            for filename in MODEL_FILENAMES.values()
        ):
            ROOT = candidate_root
            MODEL_DIR = candidate_dir
            break
    if MODEL_DIR is not None:
        break

if MODEL_DIR is None:
    MODEL_DIR = ROOT / "trained models" / "keras models"

OUTPUT_DIR = ROOT / "captured_results"
CROP_CARD_DIR = ROOT / "crop_cards"

try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROP_CARD_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    OUTPUT_DIR = Path.cwd() / "captured_results"
    CROP_CARD_DIR = Path.cwd() / "crop_cards"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROP_CARD_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# MODEL FILES
# ================================================================

MODEL_PATHS = {
    crop: MODEL_DIR / filename
    for crop, filename in MODEL_FILENAMES.items()
}


# ================================================================
# MODEL CLASS ORDERS
# ================================================================
# These must be the SAME order as the training datasets.
#
# Confirmed from the models/scripts supplied in this conversation:
#   Paddy: 7 classes
#   Cotton: 7 classes
#   Corn: 4 classes
#
# Sugarcane and Wheat .keras files only expose output count, not
# their original folder names. Keep their mapping here and change
# ONLY this section if your training report shows a different order.
# ================================================================

CLASS_NAMES = {

    "corn": [
        "Corn_Common_Rust",
        "Corn_Gray_Leaf_Spot",
        "Corn_Healthy",
        "Corn_Leaf_Blight",
    ],

    "cotton": [
        "Cotton_BacterialBlight",
        "Cotton_Healthy",
        "Cotton_Leaf_Hopper_Jassids",
        "Cotton_Leaf_Variegation",
        "Cotton_LeafCurlVirus",
        "Cotton_LeafRedding",
        "Cotton_fussarium_wilt",
    ],

    "paddy": [
        "Paddy_BacterialLeafBlight",
        "Paddy_BacterialLeafStreak",
        "Paddy_BacterialPanicleBlight",
        "Paddy_Blast",
        "Paddy_BrownSpot",
        "Paddy_Healthy",
        "Paddy_Tungro",
    ],

    "sugarcane": [
        "Sugarcane_BacterialBlight",
        "Sugarcane_Healthy",
        "Sugarcane_Mosaic",
        "Sugarcane_RedRot",
        "Sugarcane_Rust",
        "Sugarcane_YellowLeaf",
    ],

    "wheat": [
        "Wheat_Brown_Rust",
        "Wheat_Fusarium_Head_Blight",
        "Wheat_Healthy",
        "Wheat_Septoria",
    ],
}


# ================================================================
# CROP UI INFORMATION
# ================================================================

CROPS = {
    "corn": {
        "name": "Corn",
        "emoji": "🌽",
        "subtitle": "Corn disease detection",
    },
    "cotton": {
        "name": "Cotton",
        "emoji": "☁",
        "subtitle": "Cotton disease detection",
    },
    "paddy": {
        "name": "Paddy",
        "emoji": "🌾",
        "subtitle": "Rice / paddy disease detection",
    },
    "sugarcane": {
        "name": "Sugarcane",
        "emoji": "🎋",
        "subtitle": "Sugarcane disease detection",
    },
    "wheat": {
        "name": "Wheat",
        "emoji": "🌾",
        "subtitle": "Wheat disease detection",
    },
}


# ================================================================
# OFFLINE UI TRANSLATIONS
# ================================================================
# English is the default. The AI is designed so every visible UI
# string passes through tr(). Add/extend a language pack here.
# Disease names remain model class names internally and are cleaned
# for display; UI labels are translated offline.
# ================================================================

LANGUAGES = {
    "English": "en",
    "हिन्दी": "hi",
    "తెలుగు": "te",
    "தமிழ்": "ta",
    "ಕನ್ನಡ": "kn",
    "മലയാളം": "ml",
    "मराठी": "mr",
    "বাংলা": "bn",
    "ગુજરાતી": "gu",
    "ਪੰਜਾਬੀ": "pa",
    "ଓଡ଼ିଆ": "or",
}

T = {
    "en": {
        "app": "CropAI — One AI Farm Assistant",
        "welcome": "Welcome to CropAI",
        "profile": "Enter farmer details",
        "name": "Name",
        "place": "Place",
        "language": "Language",
        "continue": "Continue",
        "select_crop": "Select a crop to start AI detection",
        "start": "START AI",
        "camera": "Camera",
        "upload": "Upload Image",
        "capture": "Capture",
        "recapture": "Recapture",
        "back": "Back",
        "quit": "Quit",
        "result": "RESULT",
        "confidence": "Confidence",
        "top3": "Top 3 predictions",
        "cause": "Cause / likely factors",
        "treatment": "Management / treatment",
        "chemicals": "Common registered chemical options",
        "weather": "Field conditions",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "source_camera": "SOURCE: WEBCAM",
        "source_upload": "SOURCE: IMAGE UPLOAD",
        "save": "Save Report",
        "saved": "Report saved",
        "uncertain": "UNCERTAIN — recapture a clear single leaf",
        "healthy": "No disease class detected with high confidence.",
        "instructions": "Place ONE leaf inside the box. Use good light and focus.",
        "missing_model": "Model file not found",
        "loading": "Loading model...",
        "ready": "AI ready",
        "exit_confirm": "Exit CropAI?",
    },
    "hi": {
        "app": "क्रॉपएआई — एकीकृत कृषि सहायक",
        "welcome": "क्रॉपएआई में आपका स्वागत है",
        "profile": "किसान की जानकारी भरें",
        "name": "नाम",
        "place": "स्थान",
        "language": "भाषा",
        "continue": "आगे बढ़ें",
        "select_crop": "एआई जांच के लिए फसल चुनें",
        "start": "एआई शुरू करें",
        "camera": "कैमरा",
        "upload": "चित्र अपलोड",
        "capture": "कैप्चर",
        "recapture": "फिर से कैप्चर",
        "back": "वापस",
        "quit": "बाहर निकलें",
        "result": "परिणाम",
        "confidence": "विश्वास",
        "top3": "शीर्ष 3 अनुमान",
        "cause": "कारण / संभावित कारक",
        "treatment": "प्रबंधन / उपचार",
        "chemicals": "सामान्य पंजीकृत रासायनिक विकल्प",
        "weather": "खेत की स्थिति",
        "temperature": "तापमान",
        "humidity": "आर्द्रता",
        "source_camera": "स्रोत: वेबकैम",
        "source_upload": "स्रोत: चित्र अपलोड",
        "save": "रिपोर्ट सहेजें",
        "saved": "रिपोर्ट सहेजी गई",
        "uncertain": "अनिश्चित — साफ एकल पत्ती फिर से लें",
        "healthy": "उच्च विश्वास के साथ रोग नहीं मिला।",
        "instructions": "एक पत्ती बॉक्स में रखें। अच्छी रोशनी और फोकस रखें।",
        "missing_model": "मॉडल फ़ाइल नहीं मिली",
        "loading": "मॉडल लोड हो रहा है...",
        "ready": "एआई तैयार है",
        "exit_confirm": "क्रॉपएआई बंद करें?",
    },
    "te": {
        "app": "క్రాప్‌ఏఐ — సమగ్ర వ్యవసాయ సహాయకుడు",
        "welcome": "క్రాప్‌ఏఐకి స్వాగతం",
        "profile": "రైతు వివరాలు నమోదు చేయండి",
        "name": "పేరు",
        "place": "ప్రాంతం",
        "language": "భాష",
        "continue": "కొనసాగండి",
        "select_crop": "AI గుర్తింపుకు పంటను ఎంచుకోండి",
        "start": "AI ప్రారంభించండి",
        "camera": "కెమెరా",
        "upload": "చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "capture": "క్యాప్చర్",
        "recapture": "మళ్లీ క్యాప్చర్",
        "back": "వెనుకకు",
        "quit": "నిష్క్రమించు",
        "result": "ఫలితం",
        "confidence": "నమ్మకం",
        "top3": "టాప్ 3 అంచనాలు",
        "cause": "కారణం / సాధ్యమైన అంశాలు",
        "treatment": "నిర్వహణ / చికిత్స",
        "chemicals": "సాధారణంగా నమోదు చేసిన రసాయన ఎంపికలు",
        "weather": "పొలం పరిస్థితులు",
        "temperature": "ఉష్ణోగ్రత",
        "humidity": "తేమ",
        "source_camera": "మూలం: వెబ్‌క్యామ్",
        "source_upload": "మూలం: చిత్రం అప్‌లోడ్",
        "save": "నివేదికను సేవ్ చేయండి",
        "saved": "నివేదిక సేవ్ అయింది",
        "uncertain": "నిర్ధారణ లేదు — స్పష్టమైన ఒక్క ఆకును మళ్లీ చిత్రీకరించండి",
        "healthy": "అధిక నమ్మకంతో వ్యాధి గుర్తించబడలేదు.",
        "instructions": "ఒక ఆకును బాక్స్‌లో ఉంచండి. మంచి వెలుతురు మరియు ఫోకస్ ఉపయోగించండి.",
        "missing_model": "మోడల్ ఫైల్ కనిపించలేదు",
        "loading": "మోడల్ లోడ్ అవుతోంది...",
        "ready": "AI సిద్ధంగా ఉంది",
        "exit_confirm": "CropAI నుండి నిష్క్రమించాలా?",
    },
    "ta": {
        "app": "CropAI — ஒருங்கிணைந்த வேளாண் உதவியாளர்",
        "welcome": "CropAIக்கு வரவேற்கிறோம்",
        "profile": "விவசாயி விவரங்களை உள்ளிடவும்",
        "name": "பெயர்",
        "place": "இடம்",
        "language": "மொழி",
        "continue": "தொடரவும்",
        "select_crop": "AI கண்டறிதலுக்குப் பயிரைத் தேர்ந்தெடுக்கவும்",
        "start": "AI தொடங்கு",
        "camera": "கேமரா",
        "upload": "படத்தை பதிவேற்றவும்",
        "capture": "பிடி",
        "recapture": "மீண்டும் பிடி",
        "back": "பின்",
        "quit": "வெளியேறு",
        "result": "முடிவு",
        "confidence": "நம்பிக்கை",
        "top3": "முதல் 3 கணிப்புகள்",
        "cause": "காரணம் / சாத்திய காரணிகள்",
        "treatment": "மேலாண்மை / சிகிச்சை",
        "chemicals": "பதிவு செய்யப்பட்ட பொதுவான ரசாயன விருப்பங்கள்",
        "weather": "வயல் நிலை",
        "temperature": "வெப்பநிலை",
        "humidity": "ஈரப்பதம்",
        "source_camera": "மூலம்: வெப்கேம்",
        "source_upload": "மூலம்: படம்",
        "save": "அறிக்கையை சேமிக்கவும்",
        "saved": "அறிக்கை சேமிக்கப்பட்டது",
        "uncertain": "தெளிவில்லை — ஒரு தெளிவான இலையை மீண்டும் படம் எடுக்கவும்",
        "healthy": "அதிக நம்பிக்கையுடன் நோய் கண்டறியப்படவில்லை.",
        "instructions": "ஒரு இலையை பெட்டிக்குள் வைக்கவும். நல்ல வெளிச்சமும் ஃபோக்கஸும் பயன்படுத்தவும்.",
        "missing_model": "மாடல் கோப்பு இல்லை",
        "loading": "மாடல் ஏற்றப்படுகிறது...",
        "ready": "AI தயாராக உள்ளது",
        "exit_confirm": "CropAIயை மூடவா?",
    },
}



# ----------------------------------------------------------------
# Runtime translation support
#
# English is fully offline. For every other selected language,
# diagnostic/advisory sentences are translated through the optional
# deep-translator package and cached locally for the session.
#
# Install on UNO Q:
#   python3 -m pip install deep-translator
#
# If internet/translator access is unavailable, the UI language still
# changes, but the system safely falls back to the English source text
# instead of inventing a translation.
# ----------------------------------------------------------------

TRANSLATION_CACHE = {}

def translate_long_text(text):
    if not text:
        return text

    lang = APP.language_code

    if lang == "en":
        return text

    key = (lang, text)

    if key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[key]

    if GoogleTranslator is None:
        return text

    try:
        translated = GoogleTranslator(
            source="en",
            target=lang
        ).translate(text)

        if translated:
            TRANSLATION_CACHE[key] = translated
            return translated
    except Exception as exc:
        print("Translation fallback:", exc)

    return text

def translated_info(info):
    """Translate every natural-language diagnostic field for the selected UI language."""
    if APP.language_code == "en":
        return info

    return {
        key: translate_long_text(value)
        for key, value in info.items()
    }

# ----------------------------------------------------------------
# Full diagnostic language packs
# The underlying disease keys remain English internally. All user-visible
# diagnosis/advisory text is translated after the language is chosen.
# ----------------------------------------------------------------

DIAG_TRANSLATIONS = {
    "en": {
        "disease_information": "DISEASE INFORMATION",
        "detected": "Detected",
        "confidence": "Confidence",
        "symptoms": "SYMPTOMS / VISUAL SIGNS",
        "cause": "LIKELY CAUSE",
        "conditions": "FAVOURABLE CONDITIONS / RISK",
        "management": "MANAGEMENT / TREATMENT",
        "chemicals": "CHEMICAL / ACTIVE-INGREDIENT OPTIONS",
        "spray": "SPRAY QUANTITY",
        "monitoring": "MONITORING PLAN",
        "environment": "CURRENT DHT11 ENVIRONMENT",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "confirmation": "RECAPTURE / CONFIRMATION",
        "field_note": "FIELD NOTE",
        "use_label": "Use the current locally registered formulation and follow the exact product label.",
        "no_universal_dose": "Do not use a universal dose. The correct quantity depends on the registered formulation, crop stage, field area and label rate.",
        "context": "These sensor readings provide field context and do not replace visual scouting or agronomic confirmation.",
        "recapture_hint": "If confidence is low or symptoms are unclear, press R and capture a clearer single-leaf image.",
    },
    "hi": {
        "disease_information": "रोग की जानकारी",
        "detected": "पहचाना गया रोग",
        "confidence": "विश्वास",
        "symptoms": "लक्षण / दृश्य संकेत",
        "cause": "संभावित कारण",
        "conditions": "अनुकूल परिस्थितियाँ / जोखिम",
        "management": "प्रबंधन / उपचार",
        "chemicals": "रासायनिक / सक्रिय घटक विकल्प",
        "spray": "छिड़काव मात्रा",
        "monitoring": "निगरानी योजना",
        "environment": "वर्तमान DHT11 वातावरण",
        "temperature": "तापमान",
        "humidity": "आर्द्रता",
        "confirmation": "पुनः छवि / पुष्टि",
        "field_note": "खेत संबंधी नोट",
        "use_label": "स्थानीय रूप से पंजीकृत वर्तमान उत्पाद का उपयोग करें और उसके लेबल के अनुसार ही प्रयोग करें।",
        "no_universal_dose": "कोई सार्वभौमिक मात्रा न लें। सही मात्रा उत्पाद के फॉर्मुलेशन, फसल अवस्था, क्षेत्रफल और लेबल दर पर निर्भर करती है।",
        "context": "ये सेंसर रीडिंग खेत की स्थिति का संदर्भ देती हैं; ये दृश्य निरीक्षण या कृषि विशेषज्ञ की पुष्टि का विकल्प नहीं हैं।",
        "recapture_hint": "विश्वास कम हो या लक्षण स्पष्ट न हों तो R दबाकर एक साफ एकल पत्ती की नई छवि लें।",
    },
    "te": {
        "disease_information": "వ్యాధి సమాచారం",
        "detected": "గుర్తించిన వ్యాధి",
        "confidence": "నమ్మకం",
        "symptoms": "లక్షణాలు / దృశ్య సంకేతాలు",
        "cause": "సంభావ్య కారణం",
        "conditions": "అనుకూల పరిస్థితులు / ప్రమాదం",
        "management": "నిర్వహణ / చికిత్స",
        "chemicals": "రసాయన / క్రియాశీల పదార్థాల ఎంపికలు",
        "spray": "పిచికారీ పరిమాణం",
        "monitoring": "పర్యవేక్షణ ప్రణాళిక",
        "environment": "ప్రస్తుత DHT11 వాతావరణం",
        "temperature": "ఉష్ణోగ్రత",
        "humidity": "తేమ",
        "confirmation": "మళ్లీ చిత్రీకరణ / నిర్ధారణ",
        "field_note": "పొలం గమనిక",
        "use_label": "స్థానికంగా నమోదైన ప్రస్తుత ఉత్పత్తిని మాత్రమే ఉపయోగించి, లేబుల్ సూచనలను అనుసరించండి.",
        "no_universal_dose": "సార్వత్రిక మోతాదును ఉపయోగించవద్దు. సరైన పరిమాణం ఫార్ములేషన్, పంట దశ, విస్తీర్ణం మరియు లేబుల్ రేటుపై ఆధారపడి ఉంటుంది.",
        "context": "ఈ సెన్సార్ రీడింగ్‌లు పొల పరిస్థితులకు సందర్భాన్ని ఇస్తాయి; అవి దృశ్య పరిశీలనకు ప్రత్యామ్నాయం కావు.",
        "recapture_hint": "నమ్మకం తక్కువగా ఉంటే లేదా లక్షణాలు స్పష్టంగా లేకపోతే R నొక్కి ఒక స్పష్టమైన ఆకు చిత్రాన్ని మళ్లీ తీసుకోండి.",
    },
    "ta": {
        "disease_information": "நோய் தகவல்",
        "detected": "கண்டறியப்பட்டது",
        "confidence": "நம்பிக்கை",
        "symptoms": "அறிகுறிகள் / காட்சி குறிப்புகள்",
        "cause": "சாத்தியமான காரணம்",
        "conditions": "ஏற்ற சூழ்நிலைகள் / அபாயம்",
        "management": "மேலாண்மை / சிகிச்சை",
        "chemicals": "ரசாயன / செயற்பாட்டு மூலப்பொருள் விருப்பங்கள்",
        "spray": "தெளிப்பு அளவு",
        "monitoring": "கண்காணிப்பு திட்டம்",
        "environment": "தற்போதைய DHT11 சூழல்",
        "temperature": "வெப்பநிலை",
        "humidity": "ஈரப்பதம்",
        "confirmation": "மீண்டும் படம் / உறுதிப்படுத்தல்",
        "field_note": "வயல் குறிப்பு",
        "use_label": "உள்ளூர் பதிவு செய்யப்பட்ட தற்போதைய தயாரிப்பை மட்டும் பயன்படுத்தி லேபிள் வழிமுறைகளைப் பின்பற்றவும்.",
        "no_universal_dose": "பொதுவான ஒரே அளவை பயன்படுத்த வேண்டாம். சரியான அளவு தயாரிப்பு வடிவம், பயிர் நிலை, பரப்பளவு மற்றும் லேபிள் விகிதத்தைப் பொறுத்தது.",
        "context": "இந்த சென்சார் அளவீடுகள் வயல் நிலைக்கு சூழலை வழங்குகின்றன; அவை நேரடி கண்காணிப்பிற்கு மாற்றாகாது.",
        "recapture_hint": "நம்பிக்கை குறைந்தால் அல்லது அறிகுறிகள் தெளிவாக இல்லாவிட்டால் R அழுத்தி தெளிவான ஒரு இலை படத்தை மீண்டும் எடுக்கவும்.",
    },
    "kn": {
        "disease_information": "ರೋಗ ಮಾಹಿತಿ",
        "detected": "ಗುರುತಿಸಿದ ರೋಗ",
        "confidence": "ನಂಬಿಕೆ",
        "symptoms": "ಲಕ್ಷಣಗಳು / ದೃಶ್ಯ ಸೂಚನೆಗಳು",
        "cause": "ಸಂಭಾವ್ಯ ಕಾರಣ",
        "conditions": "ಅನುಕೂಲ ಪರಿಸ್ಥಿತಿಗಳು / ಅಪಾಯ",
        "management": "ನಿರ್ವಹಣೆ / ಚಿಕಿತ್ಸೆ",
        "chemicals": "ರಾಸಾಯನಿಕ / ಸಕ್ರಿಯ ಘಟಕ ಆಯ್ಕೆಗಳು",
        "spray": "ಸಿಂಪಡಣೆ ಪ್ರಮಾಣ",
        "monitoring": "ಮೇಲ್ವಿಚಾರಣೆ ಯೋಜನೆ",
        "environment": "ಪ್ರಸ್ತುತ DHT11 ಪರಿಸರ",
        "temperature": "ತಾಪಮಾನ",
        "humidity": "ತೇವಾಂಶ",
        "confirmation": "ಮತ್ತೆ ಚಿತ್ರ / ದೃಢೀಕರಣ",
        "field_note": "ಹೊಲದ ಟಿಪ್ಪಣಿ",
        "use_label": "ಸ್ಥಳೀಯವಾಗಿ ನೋಂದಾಯಿತ ಪ್ರಸ್ತುತ ಉತ್ಪನ್ನವನ್ನು ಮಾತ್ರ ಬಳಸಿ ಮತ್ತು ಲೇಬಲ್ ಸೂಚನೆಗಳನ್ನು ಪಾಲಿಸಿ.",
        "no_universal_dose": "ಒಂದೇ ಸಾಮಾನ್ಯ ಪ್ರಮಾಣವನ್ನು ಬಳಸಬೇಡಿ. ಸರಿಯಾದ ಪ್ರಮಾಣವು ಫಾರ್ಮುಲೇಶನ್, ಬೆಳೆ ಹಂತ, ಜಮೀನು ವಿಸ್ತೀರ್ಣ ಮತ್ತು ಲೇಬಲ್ ದರವನ್ನು ಅವಲಂಬಿಸುತ್ತದೆ.",
        "context": "ಈ ಸೆನ್ಸರ್ ಅಳತೆಗಳು ಹೊಲದ ಪರಿಸ್ಥಿತಿಗೆ ಸಂದರ್ಭ ನೀಡುತ್ತವೆ; ಅವು ದೃಶ್ಯ ಪರಿಶೀಲನೆಗೆ ಪರ್ಯಾಯವಲ್ಲ.",
        "recapture_hint": "ನಂಬಿಕೆ ಕಡಿಮೆಯಿದ್ದರೆ ಅಥವಾ ಲಕ್ಷಣಗಳು ಸ್ಪಷ್ಟವಾಗದಿದ್ದರೆ R ಒತ್ತಿ ಸ್ಪಷ್ಟವಾದ ಒಂದು ಎಲೆಯ ಚಿತ್ರವನ್ನು ಮತ್ತೆ ತೆಗೆದುಕೊಳ್ಳಿ.",
    },
    "ml": {
        "disease_information": "രോഗ വിവരം",
        "detected": "കണ്ടെത്തിയത്",
        "confidence": "വിശ്വാസനില",
        "symptoms": "ലക്ഷണങ്ങൾ / ദൃശ്യ സൂചനകൾ",
        "cause": "സാധ്യതയുള്ള കാരണം",
        "conditions": "അനുകൂല സാഹചര്യം / അപകടസാധ്യത",
        "management": "നിയന്ത്രണം / ചികിത്സ",
        "chemicals": "രാസ / സജീവ ഘടക ഓപ്ഷനുകൾ",
        "spray": "തളിക്കൽ അളവ്",
        "monitoring": "നിരീക്ഷണ പദ്ധതി",
        "environment": "നിലവിലെ DHT11 പരിസ്ഥിതി",
        "temperature": "താപനില",
        "humidity": "ആർദ്രത",
        "confirmation": "വീണ്ടും ചിത്രം / സ്ഥിരീകരണം",
        "field_note": "കൃഷിയിട കുറിപ്പ്",
        "use_label": "പ്രാദേശികമായി രജിസ്റ്റർ ചെയ്ത നിലവിലെ ഉൽപ്പന്നം മാത്രം ഉപയോഗിച്ച് ലേബൽ നിർദ്ദേശങ്ങൾ പാലിക്കുക.",
        "no_universal_dose": "ഒറ്റ പൊതുവായ അളവ് ഉപയോഗിക്കരുത്. ശരിയായ അളവ് ഫോർമുലേഷൻ, വിളയുടെ ഘട്ടം, വിസ്തീർണ്ണം, ലേബൽ നിരക്ക് എന്നിവയെ ആശ്രയിച്ചിരിക്കും.",
        "context": "ഈ സെൻസർ റീഡിംഗുകൾ കൃഷിയിട സാഹചര്യത്തിന് പശ്ചാത്തലം നൽകുന്നു; അവ ദൃശ്യ പരിശോധനയ്ക്ക് പകരമല്ല.",
        "recapture_hint": "വിശ്വാസനില കുറവായാൽ അല്ലെങ്കിൽ ലക്ഷണങ്ങൾ വ്യക്തമല്ലെങ്കിൽ R അമർത്തി വ്യക്തമായ ഒരു ഇലയുടെ ചിത്രം വീണ്ടും എടുക്കുക.",
    },
    "mr": {
        "disease_information": "रोग माहिती",
        "detected": "ओळखलेला रोग",
        "confidence": "विश्वास",
        "symptoms": "लक्षणे / दृश्य संकेत",
        "cause": "संभाव्य कारण",
        "conditions": "अनुकूल परिस्थिती / धोका",
        "management": "व्यवस्थापन / उपचार",
        "chemicals": "रासायनिक / सक्रिय घटक पर्याय",
        "spray": "फवारणीचे प्रमाण",
        "monitoring": "निगराणी योजना",
        "environment": "सध्याचे DHT11 वातावरण",
        "temperature": "तापमान",
        "humidity": "आर्द्रता",
        "confirmation": "पुन्हा छायाचित्र / पुष्टी",
        "field_note": "शेत नोंद",
        "use_label": "स्थानिक नोंदणीकृत सध्याचे उत्पादन वापरा आणि लेबलच्या सूचनांचे पालन करा.",
        "no_universal_dose": "सर्वांसाठी एकच डोस वापरू नका. योग्य प्रमाण फॉर्म्युलेशन, पीक अवस्था, क्षेत्रफळ आणि लेबल दरावर अवलंबून असते.",
        "context": "हे सेन्सर रीडिंग्स शेताच्या स्थितीचा संदर्भ देतात; ते प्रत्यक्ष निरीक्षणाचा पर्याय नाहीत.",
        "recapture_hint": "विश्वास कमी असल्यास किंवा लक्षणे स्पष्ट नसल्यास R दाबून स्वच्छ एकाच पानाचे छायाचित्र पुन्हा घ्या.",
    },
    "bn": {
        "disease_information": "রোগের তথ্য",
        "detected": "শনাক্ত রোগ",
        "confidence": "বিশ্বাসযোগ্যতা",
        "symptoms": "উপসর্গ / দৃশ্যমান লক্ষণ",
        "cause": "সম্ভাব্য কারণ",
        "conditions": "অনুকূল পরিবেশ / ঝুঁকি",
        "management": "ব্যবস্থাপনা / চিকিৎসা",
        "chemicals": "রাসায়নিক / সক্রিয় উপাদানের বিকল্প",
        "spray": "স্প্রে পরিমাণ",
        "monitoring": "পর্যবেক্ষণ পরিকল্পনা",
        "environment": "বর্তমান DHT11 পরিবেশ",
        "temperature": "তাপমাত্রা",
        "humidity": "আর্দ্রতা",
        "confirmation": "পুনরায় ছবি / নিশ্চিতকরণ",
        "field_note": "ক্ষেত্র নোট",
        "use_label": "স্থানীয়ভাবে নিবন্ধিত বর্তমান পণ্য ব্যবহার করুন এবং লেবেলের নির্দেশনা মেনে চলুন।",
        "no_universal_dose": "একটি সাধারণ ডোজ ব্যবহার করবেন না। সঠিক পরিমাণ ফর্মুলেশন, ফসলের পর্যায়, জমির আয়তন এবং লেবেল হারের উপর নির্ভর করে।",
        "context": "এই সেন্সর রিডিংগুলি ক্ষেতের পরিবেশ বোঝাতে সাহায্য করে; এগুলি দৃশ্যমান পর্যবেক্ষণের বিকল্প নয়।",
        "recapture_hint": "বিশ্বাসযোগ্যতা কম হলে বা উপসর্গ পরিষ্কার না হলে R চাপুন এবং পরিষ্কার একটি পাতার ছবি আবার নিন।",
    },
    "gu": {
        "disease_information": "રોગની માહિતી",
        "detected": "ઓળખાયેલ રોગ",
        "confidence": "વિશ્વાસ",
        "symptoms": "લક્ષણો / દૃશ્ય સંકેતો",
        "cause": "સંભવિત કારણ",
        "conditions": "અનુકૂળ પરિસ્થિતિઓ / જોખમ",
        "management": "વ્યવસ્થાપન / સારવાર",
        "chemicals": "રાસાયણિક / સક્રિય ઘટક વિકલ્પો",
        "spray": "છંટકાવનું પ્રમાણ",
        "monitoring": "નિરીક્ષણ યોજના",
        "environment": "વર્તમાન DHT11 પરિસ્થિતિ",
        "temperature": "તાપમાન",
        "humidity": "ભેજ",
        "confirmation": "ફરી છબી / પુષ્ટિ",
        "field_note": "ખેતરની નોંધ",
        "use_label": "સ્થાનિક રીતે નોંધાયેલ વર્તમાન ઉત્પાદનનો જ ઉપયોગ કરો અને લેબલની સૂચનાઓનું પાલન કરો.",
        "no_universal_dose": "એક સામાન્ય ડોઝનો ઉપયોગ ન કરો. યોગ્ય પ્રમાણ ફોર્મ્યુલેશન, પાકની અવસ્થા, વિસ્તાર અને લેબલ દર પર આધારિત છે.",
        "context": "આ સેન્સર વાંચનો ખેતરની પરિસ્થિતિ માટે સંદર્ભ આપે છે; તે દૃશ્ય નિરીક્ષણનો વિકલ્પ નથી.",
        "recapture_hint": "વિશ્વાસ ઓછો હોય અથવા લક્ષણો સ્પષ્ટ ન હોય તો R દબાવીને એક સ્પષ્ટ પાનની છબી ફરી લો.",
    },
    "pa": {
        "disease_information": "ਬਿਮਾਰੀ ਦੀ ਜਾਣਕਾਰੀ",
        "detected": "ਪਛਾਣੀ ਬਿਮਾਰੀ",
        "confidence": "ਭਰੋਸਾ",
        "symptoms": "ਲੱਛਣ / ਦ੍ਰਿਸ਼ਟੀ ਸੰਕੇਤ",
        "cause": "ਸੰਭਾਵਿਤ ਕਾਰਨ",
        "conditions": "ਅਨੁਕੂਲ ਹਾਲਾਤ / ਖਤਰਾ",
        "management": "ਪ੍ਰਬੰਧਨ / ਇਲਾਜ",
        "chemicals": "ਰਸਾਇਣਕ / ਸਰਗਰਮ ਤੱਤਾਂ ਦੇ ਵਿਕਲਪ",
        "spray": "ਛਿੜਕਾਅ ਮਾਤਰਾ",
        "monitoring": "ਨਿਗਰਾਨੀ ਯੋਜਨਾ",
        "environment": "ਮੌਜੂਦਾ DHT11 ਵਾਤਾਵਰਣ",
        "temperature": "ਤਾਪਮਾਨ",
        "humidity": "ਨਮੀ",
        "confirmation": "ਮੁੜ ਤਸਵੀਰ / ਪੁਸ਼ਟੀ",
        "field_note": "ਖੇਤ ਨੋਟ",
        "use_label": "ਸਥਾਨਕ ਤੌਰ 'ਤੇ ਰਜਿਸਟਰ ਕੀਤਾ ਮੌਜੂਦਾ ਉਤਪਾਦ ਹੀ ਵਰਤੋ ਅਤੇ ਲੇਬਲ ਦੀਆਂ ਹਦਾਇਤਾਂ ਮੰਨੋ।",
        "no_universal_dose": "ਇੱਕੋ ਸਧਾਰਣ ਖੁਰਾਕ ਨਾ ਵਰਤੋ। ਸਹੀ ਮਾਤਰਾ ਫਾਰਮੂਲੇਸ਼ਨ, ਫਸਲ ਦੀ ਅਵਸਥਾ, ਖੇਤਰਫਲ ਅਤੇ ਲੇਬਲ ਦਰ 'ਤੇ ਨਿਰਭਰ ਕਰਦੀ ਹੈ।",
        "context": "ਇਹ ਸੈਂਸਰ ਰੀਡਿੰਗ ਖੇਤ ਦੀ ਸਥਿਤੀ ਲਈ ਸੰਦਰਭ ਦਿੰਦੀਆਂ ਹਨ; ਇਹ ਦ੍ਰਿਸ਼ਟੀ ਜਾਂਚ ਦਾ ਬਦਲ ਨਹੀਂ ਹਨ।",
        "recapture_hint": "ਜੇ ਭਰੋਸਾ ਘੱਟ ਹੋਵੇ ਜਾਂ ਲੱਛਣ ਸਪਸ਼ਟ ਨਾ ਹੋਣ ਤਾਂ R ਦਬਾ ਕੇ ਇੱਕ ਸਾਫ਼ ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਮੁੜ ਲਵੋ।",
    },
    "or": {
        "disease_information": "ରୋଗ ସୂଚନା",
        "detected": "ଚିହ୍ନଟ ରୋଗ",
        "confidence": "ବିଶ୍ୱାସ",
        "symptoms": "ଲକ୍ଷଣ / ଦୃଶ୍ୟ ସଙ୍କେତ",
        "cause": "ସମ୍ଭାବ୍ୟ କାରଣ",
        "conditions": "ଅନୁକୂଳ ପରିସ୍ଥିତି / ଝୁମ୍ପ",
        "management": "ପରିଚାଳନା / ଚିକିତ୍ସା",
        "chemicals": "ରାସାୟନିକ / ସକ୍ରିୟ ଉପାଦାନ ବିକଳ୍ପ",
        "spray": "ସ୍ପ୍ରେ ପରିମାଣ",
        "monitoring": "ନିରୀକ୍ଷଣ ଯୋଜନା",
        "environment": "ବର୍ତ୍ତମାନ DHT11 ପରିବେଶ",
        "temperature": "ତାପମାତ୍ରା",
        "humidity": "ଆର୍ଦ୍ରତା",
        "confirmation": "ପୁନଃ ଛବି / ନିଶ୍ଚିତକରଣ",
        "field_note": "କ୍ଷେତ୍ର ଟୀକା",
        "use_label": "ସ୍ଥାନୀୟ ଭାବେ ପଞ୍ଜୀକୃତ ବର୍ତ୍ତମାନ ଉତ୍ପାଦ ବ୍ୟବହାର କରନ୍ତୁ ଏବଂ ଲେବେଲ ନିର୍ଦ୍ଦେଶ ମାନନ୍ତୁ।",
        "no_universal_dose": "ଏକ ସାର୍ବଜନୀନ ଡୋଜ ବ୍ୟବହାର କରନ୍ତୁ ନାହିଁ। ସଠିକ ପରିମାଣ ଫର୍ମୁଲେସନ, ଫସଲ ଅବସ୍ଥା, କ୍ଷେତ୍ର ଆକାର ଓ ଲେବେଲ ଦର ଉପରେ ନିର୍ଭର କରେ।",
        "context": "ଏହି ସେନ୍ସର ରିଡିଂଗୁଡିକ କ୍ଷେତ୍ର ପରିସ୍ଥିତିର ସନ୍ଦର୍ଭ ଦିଏ; ଏଗୁଡିକ ଦୃଶ୍ୟ ନିରୀକ୍ଷଣର ପର୍ଯ୍ୟାୟ ନୁହେଁ।",
        "recapture_hint": "ବିଶ୍ୱାସ କମ୍ ଥିଲେ କିମ୍ବା ଲକ୍ଷଣ ସ୍ପଷ୍ଟ ନ ଥିଲେ R ଦବାଇ ଏକ ସ୍ପଷ୍ଟ ପତ୍ରର ଛବି ପୁଣି ନିଅନ୍ତୁ।",
    },
}


# For languages not yet given a full offline pack, English remains
# the safe fallback rather than showing incorrect machine translations.
# The language is still stored in the report and can be expanded later.


def tr(key):
    """Translate UI text using the selected offline language pack."""
    lang = APP.language_code
    return T.get(lang, T["en"]).get(key, T["en"].get(key, key))


# ================================================================
# DISEASE KNOWLEDGE
# ================================================================
# This is deliberately conservative. Product formulation and dose
# vary by country, crop stage, disease severity and label. The UI
# therefore tells the farmer to use a locally registered product
# exactly according to its current label instead of hard-coding an
# unsafe universal grams/ml dose.
# ================================================================

DISEASE_INFO = {

    # ============================================================
    # PADDY
    # ============================================================

    "Paddy_BacterialLeafBlight": {
        "symptoms": "Water-soaked to yellow lesions usually begin near leaf tips or margins and expand along the leaf. Severely affected leaves can wilt and dry.",
        "cause": "Bacterial disease associated with Xanthomonas oryzae; spread is favoured by rain, splashing water, wounds and susceptible plants.",
        "conditions": "Warm, humid and wet weather, prolonged leaf wetness and excessive nitrogen can increase risk.",
        "management": "Use clean seed/planting material, balanced nitrogen, resistant varieties where available, field sanitation and careful irrigation. Avoid unnecessary leaf wetness.",
        "chemicals": "Use only locally registered rice bactericide products. Copper-based or other registered bactericide options may be recommended by local authorities.",
        "monitoring": "Inspect new leaves and neighbouring plants every few days after rain or high-humidity periods. Confirm unusual symptoms before chemical treatment."
    },

    "Paddy_BacterialLeafStreak": {
        "symptoms": "Narrow water-soaked streaks develop between leaf veins and may become translucent or yellow-brown as lesions enlarge.",
        "cause": "Bacterial disease affecting rice leaves; infection is favoured by wet foliage and plant injury.",
        "conditions": "Warm, humid weather, rain and prolonged leaf wetness favour spread.",
        "management": "Use clean seed, balanced fertilizer, sanitation and avoid unnecessary leaf wetness or excessive nitrogen.",
        "chemicals": "Use only locally registered rice bactericide products and follow the current formulation-specific label.",
        "monitoring": "Check surrounding leaves for increasing translucent streaks and compare newly emerging leaves with older leaves."
    },

    "Paddy_BacterialPanicleBlight": {
        "symptoms": "Panicles or developing grains can show discoloration, poor grain filling and blighting of affected tissues.",
        "cause": "Bacterial infection associated with panicle-stage disease development under favourable humid conditions.",
        "conditions": "Warm, humid and wet conditions around panicle development can increase risk.",
        "management": "Use clean seed, balanced nutrition, field sanitation and timely scouting around flowering and grain filling.",
        "chemicals": "Use only products specifically registered/recommended for the crop and target disease in your region.",
        "monitoring": "Inspect panicles during flowering and grain development, especially after prolonged wet weather."
    },

    "Paddy_Blast": {
        "symptoms": "Typical leaf lesions are spindle/diamond-shaped with grey centres and darker margins; neck and panicle infections can cause severe yield loss.",
        "cause": "Fungal disease caused by Magnaporthe oryzae and related blast populations.",
        "conditions": "High humidity, prolonged leaf wetness, dense canopy, susceptible varieties and excessive nitrogen favour development.",
        "management": "Use resistant varieties where available, balanced nitrogen, good field ventilation, careful water management and early scouting.",
        "chemicals": "ICAR advisory material includes tricyclazole for seed treatment and propiconazole for spray management of rice blast; use only current locally registered products and label directions.",
        "monitoring": "Inspect leaves, nodes and panicles. Recheck 7–14 days after an approved intervention or sooner if symptoms spread rapidly."
    },

    "Paddy_BrownSpot": {
        "symptoms": "Small brown circular to oval spots can enlarge into darker lesions, commonly starting on older leaves and increasing under crop stress.",
        "cause": "Fungal disease associated with Bipolaris oryzae; weak plants and nutritional stress can increase susceptibility.",
        "conditions": "Warm humid conditions combined with nutrient imbalance, water stress or poor plant vigor can increase risk.",
        "management": "Maintain balanced nutrition, improve plant vigor, use clean seed and avoid prolonged drought or other crop stress.",
        "chemicals": "Use a locally registered rice fungicide only when recommended for Brown Spot and follow the exact current product label.",
        "monitoring": "Compare lesion number and size on new leaves over time. Watch crop vigor and nutrition as well as disease symptoms."
    },

    "Paddy_Healthy": {
        "symptoms": "No strong disease pattern was selected by the model at the time of analysis.",
        "cause": "The model did not identify one of the trained disease classes with sufficient confidence.",
        "conditions": "Healthy appearance should still be interpreted together with field temperature, humidity, irrigation, nutrition and pest pressure.",
        "management": "Continue routine scouting, balanced irrigation and nutrition, field sanitation and preventive crop management.",
        "chemicals": "No disease chemical treatment is recommended solely from a Healthy prediction.",
        "monitoring": "Repeat inspection when symptoms first appear or after major changes in weather, irrigation or pest pressure."
    },

    "Paddy_Tungro": {
        "symptoms": "Typical signs include yellow to orange discoloration, stunting and reduced tillering; patterns can vary with infection stage.",
        "cause": "Viral disease mainly transmitted by green leafhopper vectors.",
        "conditions": "Vector activity, susceptible varieties and infected planting material contribute to spread.",
        "management": "Use resistant/tolerant varieties where available, healthy planting material, vector monitoring, rogueing where recommended and integrated pest management.",
        "chemicals": "There is no direct curative chemical for the virus. Vector control should use only locally registered insecticides when economically justified and label-approved.",
        "monitoring": "Check nearby plants and new growth for yellowing and stunting; monitor vector populations."
    },

    # ============================================================
    # COTTON
    # ============================================================

    "Cotton_BacterialBlight": {
        "symptoms": "Angular or water-soaked leaf lesions may darken and expand; petiole and boll symptoms can occur in severe cases.",
        "cause": "Bacterial disease associated with infected seed, plant debris and wet conditions.",
        "conditions": "Warm, humid and rainy weather can favour infection and spread.",
        "management": "Use clean seed, sanitation, balanced nutrition, resistant cultivars where available and avoid unnecessary overhead wetting.",
        "chemicals": "Use only locally registered cotton bactericide/copper-based products where recommended by local guidance.",
        "monitoring": "Inspect leaves and bolls after rainfall and monitor nearby plants for expanding lesions."
    },

    "Cotton_Healthy": {
        "symptoms": "No trained disease/pest pattern was selected with high confidence.",
        "cause": "The model did not identify a specific Cotton disease class.",
        "conditions": "Continue monitoring because pests, nutrient stress and disease may develop after the current image.",
        "management": "Maintain balanced irrigation, nutrition, field sanitation and integrated pest management.",
        "chemicals": "No chemical treatment should be triggered solely from a Healthy prediction.",
        "monitoring": "Regularly inspect young leaves, leaf undersides, squares and bolls."
    },

    "Cotton_Leaf_Hopper_Jassids": {
        "symptoms": "Feeding damage can cause yellowing, curling, marginal reddening and a scorched appearance on leaf margins; insects are often found on leaf undersides.",
        "cause": "Sap-feeding by cotton jassids/leafhoppers.",
        "conditions": "Warm crop conditions and active sucking-pest populations favour damage.",
        "management": "Scout leaf undersides, use integrated pest management and act according to locally established economic thresholds.",
        "chemicals": "ICAR-CICR advisory material lists active ingredients such as flonicamid, dinotefuran and other registered sucking-pest options for jassid management; use only a currently registered product and label rate.",
        "monitoring": "Count pests on representative leaves and track whether feeding damage is increasing before repeating applications."
    },

    "Cotton_Leaf_Variegation": {
        "symptoms": "Irregular light and dark green patterns or chlorotic patches can appear over the leaf surface.",
        "cause": "Variegation can reflect nutritional imbalance, stress, pest injury, virus-like symptoms or other non-specific causes.",
        "conditions": "Stress, nutrient imbalance, root problems or environmental fluctuations may contribute.",
        "management": "Inspect roots, irrigation, nutrition and pest pressure before assuming a specific pathogen.",
        "chemicals": "Do not apply a disease chemical solely from this visual class; address the underlying stressor after field diagnosis.",
        "monitoring": "Compare new leaves with older leaves and check whether the pattern is spreading across the field."
    },

    "Cotton_LeafCurlVirus": {
        "symptoms": "Leaves can curl upward or downward, become smaller and show vein thickening, enations or reduced growth depending on infection stage.",
        "cause": "Cotton leaf curl disease caused by a virus complex and associated with whitefly transmission.",
        "conditions": "Whitefly activity and favourable warm conditions can increase field spread.",
        "management": "Use resistant/tolerant varieties where available and integrate whitefly monitoring and management.",
        "chemicals": "No insecticide cures the virus itself. Registered vector-control products may be used only when justified and according to the current label.",
        "monitoring": "Inspect young leaves and leaf undersides and monitor whitefly abundance on nearby plants."
    },

    "Cotton_LeafRedding": {
        "symptoms": "Leaves may develop red, bronze or purplish coloration, often starting on older leaves and sometimes progressing upward.",
        "cause": "Often associated with nutrient imbalance, water stress, root stress or heavy fruit load rather than a single pathogen.",
        "conditions": "Water stress, nutrient imbalance and certain weather/soil conditions can contribute.",
        "management": "Check soil moisture, nutrition, root health and crop load before choosing a chemical treatment.",
        "chemicals": "Chemical disease control is not automatically indicated; correct the underlying stress and follow local agronomic guidance.",
        "monitoring": "Observe whether reddening improves after correcting moisture/nutrition and whether new leaves remain affected."
    },

    "Cotton_fussarium_wilt": {
        "symptoms": "Yellowing, wilting and vascular browning can develop, with affected plants showing reduced vigor and sometimes uneven field patches.",
        "cause": "Soil-borne Fusarium infection associated with vascular wilt.",
        "conditions": "Infected soil/planting material, susceptible varieties and plant stress can increase incidence.",
        "management": "Use healthy planting material, resistant/tolerant varieties where available, sanitation, crop rotation and good field management.",
        "chemicals": "Use only locally registered seed-treatment or soil-management products recommended for cotton wilt; no universal curative spray rate should be assumed.",
        "monitoring": "Inspect plant bases and vascular tissue in suspicious plants and map recurring field patches."
    },

    # ============================================================
    # CORN
    # ============================================================

    "Corn_Common_Rust": {
        "symptoms": "Small reddish-brown to cinnamon rust pustules form on leaves and release spores when rubbed.",
        "cause": "Fungal rust disease caused by Puccinia species affecting maize.",
        "conditions": "Leaf wetness and suitable cool-to-moderate temperatures can favour infection.",
        "management": "Use resistant hybrids where available, maintain crop vigor and monitor early lesions.",
        "chemicals": "Where locally registered and economically justified, corn fungicides from triazole or strobilurin groups may be used according to the current label.",
        "monitoring": "Inspect upper and middle leaves and record whether pustules are increasing or spreading."
    },

    "Corn_Gray_Leaf_Spot": {
        "symptoms": "Long rectangular grey-to-tan lesions develop between leaf veins and can merge into larger necrotic areas.",
        "cause": "Fungal foliar disease caused by Cercospora species.",
        "conditions": "Warm humid weather, prolonged leaf wetness, dense canopy and susceptible hybrids favour risk.",
        "management": "Use resistant/tolerant hybrids, residue and rotation practices where appropriate and avoid unnecessary canopy stress.",
        "chemicals": "Use only locally registered corn fungicides; product choice and timing should follow the label and local recommendations.",
        "monitoring": "Pay particular attention to lower and middle leaves before lesions move upward toward the ear leaf."
    },

    "Corn_Healthy": {
        "symptoms": "No trained Corn disease pattern was selected with sufficient confidence.",
        "cause": "The model did not identify one of the trained disease classes strongly enough.",
        "conditions": "Field conditions should still be assessed because environmental stress can precede visible disease.",
        "management": "Continue scouting, balanced nutrition, irrigation and integrated crop management.",
        "chemicals": "No disease chemical treatment should be triggered solely from a Healthy prediction.",
        "monitoring": "Repeat inspection if lesions, discoloration or pest damage appear."
    },

    "Corn_Leaf_Blight": {
        "symptoms": "Elongated tan, brown or grey lesions can develop and expand along the leaf; severe infection may reduce green leaf area.",
        "cause": "Fungal leaf-blight complex; the exact causal species depends on the disease presentation and region.",
        "conditions": "Warm humid weather and prolonged leaf wetness generally increase risk.",
        "management": "Use resistant hybrids, residue management, rotation where appropriate and timely scouting.",
        "chemicals": "Use only a locally registered corn fungicide appropriate for the diagnosed disease and crop stage.",
        "monitoring": "Monitor the ear leaf and leaves above/below it because loss of green area there is important for yield."
    },

    # ============================================================
    # SUGARCANE
    # ============================================================

    "Sugarcane_BacterialBlight": {
        "symptoms": "Leaf streaking, water-soaked areas and necrotic patches can appear depending on the causal bacterium and disease stage.",
        "cause": "Bacterial infection affecting leaves or stalk tissue.",
        "conditions": "Warm, humid and wet conditions may favour bacterial spread.",
        "management": "Use healthy planting material, field sanitation, balanced nutrition and good drainage.",
        "chemicals": "Use only locally registered sugarcane bactericide products where such treatment is recommended.",
        "monitoring": "Inspect leaves and stalks in affected blocks and watch for rapid spread after wet weather."
    },

    "Sugarcane_Healthy": {
        "symptoms": "No trained Sugarcane disease class was selected with high confidence.",
        "cause": "The model did not identify a specific disease pattern.",
        "conditions": "Crop health should still be checked against moisture, drainage, nutrition and pest pressure.",
        "management": "Maintain healthy seed cane, balanced nutrient supply, irrigation and drainage.",
        "chemicals": "No chemical treatment should be triggered solely from a Healthy prediction.",
        "monitoring": "Regularly inspect leaves, stalks and planting material for early disease signs."
    },

    "Sugarcane_Mosaic": {
        "symptoms": "Leaves show contrasting light and dark green mosaic or mottled patterns; severe cases may show reduced vigor.",
        "cause": "Viral disease associated with sugarcane mosaic viruses and vegetative propagation.",
        "conditions": "Infected planting material and vector activity can spread disease.",
        "management": "Use healthy seed cane, resistant/tolerant varieties, sanitation and rogue severely affected plants where recommended.",
        "chemicals": "There is no direct curative chemical for the virus; manage planting material and vectors through integrated methods.",
        "monitoring": "Check new leaves and neighbouring stools for expanding mosaic symptoms."
    },

    "Sugarcane_RedRot": {
        "symptoms": "Internal stalk tissues show characteristic reddish discoloration, often with pale cross bands; leaves can wilt and dry as disease progresses.",
        "cause": "Fungal disease associated with red rot pathogens of sugarcane.",
        "conditions": "Infected setts, poor drainage, warm humid weather and susceptible varieties increase risk.",
        "management": "Use healthy disease-free setts, resistant varieties, sanitation, good drainage and avoid carrying infected planting material.",
        "chemicals": "Use locally recommended planting-material treatments only where registered. ICAR material also reports Trichoderma-based management options for red rot; follow local recommendations.",
        "monitoring": "Split suspect stalks to inspect internal discoloration and map disease patches for sanitation."
    },

    "Sugarcane_Rust": {
        "symptoms": "Small rust-coloured pustules develop on leaves, often with surrounding yellowing or orange-brown areas.",
        "cause": "Fungal rust disease affecting sugarcane leaves.",
        "conditions": "Humid conditions and susceptible varieties favour disease development.",
        "management": "Use resistant varieties where available, maintain crop vigor and monitor disease early.",
        "chemicals": "Use only locally registered sugarcane fungicides according to the current label.",
        "monitoring": "Inspect older and middle leaves and observe whether pustules are moving to upper canopy leaves."
    },

    "Sugarcane_YellowLeaf": {
        "symptoms": "Leaf midrib yellowing, especially on older leaves, can be accompanied by reduced vigor and varietal degeneration.",
        "cause": "Yellow leaf disease is associated with Sugarcane yellow leaf virus and can spread through infected setts and aphid vectors.",
        "conditions": "Infected planting material and vector activity increase spread risk.",
        "management": "Use healthy disease-free seed cane, tolerant varieties where available and integrated vector/seed health management.",
        "chemicals": "There is no direct chemical cure for the virus; vector management should follow local registered recommendations.",
        "monitoring": "Inspect the underside/midrib area of mature leaves and compare infected stools with healthy adjacent stools."
    },

    # ============================================================
    # WHEAT
    # ============================================================

    "Wheat_Brown_Rust": {
        "symptoms": "Small orange-brown to reddish-brown rust pustules appear mainly on leaves and may coalesce as disease progresses.",
        "cause": "Fungal rust disease caused by Puccinia species.",
        "conditions": "Suitable temperature, moisture and susceptible varieties favour rust development.",
        "management": "Use resistant varieties where available, scout regularly and protect upper leaves when disease risk is high.",
        "chemicals": "Use only a locally registered wheat fungicide for rust and follow the current label and crop-stage recommendation.",
        "monitoring": "Monitor the flag leaf and upper canopy because disease there has greater yield impact."
    },

    "Wheat_Fusarium_Head_Blight": {
        "symptoms": "Individual spikelets may bleach prematurely; infected heads can show pink/orange fungal growth under humid conditions.",
        "cause": "Fungal disease complex associated with Fusarium species affecting wheat heads.",
        "conditions": "Wet, humid conditions around flowering increase risk.",
        "management": "Use resistant/tolerant varieties where available, crop rotation and timely field scouting around flowering.",
        "chemicals": "Use only locally registered wheat fungicides specifically approved for head blight and follow label timing.",
        "monitoring": "Inspect heads during flowering and early grain development and monitor disease after wet weather."
    },

    "Wheat_Healthy": {
        "symptoms": "No trained wheat disease pattern was selected with high confidence.",
        "cause": "The model did not identify a specific disease class.",
        "conditions": "Continue field assessment because rusts and foliar diseases can develop rapidly under favourable weather.",
        "management": "Maintain balanced nutrition, irrigation and integrated pest/disease scouting.",
        "chemicals": "No chemical disease treatment should be triggered solely from a Healthy prediction.",
        "monitoring": "Recheck the crop after rainfall, prolonged dew or visible colour/lesion changes."
    },

    "Wheat_Septoria": {
        "symptoms": "Small dark specks and tan-brown lesions may develop on leaves, often beginning on lower leaves and progressing upward.",
        "cause": "Septoria foliar disease complex affecting wheat.",
        "conditions": "Prolonged leaf wetness, rainfall and dense canopy conditions favour development.",
        "management": "Use resistant varieties where available, crop rotation, residue management and timely disease scouting.",
        "chemicals": "Use only registered wheat fungicides suitable for Septoria and follow the current product label.",
        "monitoring": "Watch lower leaves first, then monitor progression toward the flag leaf."
    }
}


# ================================================================
# APPLICATION STATE
# ================================================================

class State:
    name = ""
    place = ""
    language = "English"
    language_code = "en"

    crop = None
    model = None

    temperature = None
    humidity = None

    current_image = None
    current_source = None

    result_name = None
    confidence = 0.0
    top_predictions = []


APP = State()


# ================================================================
# DHT11 SERIAL READER
# ================================================================

class DHTBridgeReader:
    """
    UNO Q DHT11 reader using Arduino Router Bridge.

    MCU side exposes:
      get_dht_temperature()
      get_dht_humidity()

    Linux side calls those RPC methods through:
      /var/run/arduino-router.sock

    This avoids relying on /dev/ttyACM0, which is not the correct
    Linux-to-MCU interface for the UNO Q Bridge architecture.
    """

    SOCKET_PATH = "/var/run/arduino-router.sock"

    def __init__(self):
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.request_ids = itertools.count(1)

    def start(self):
        if msgpack is None:
            print(
                "DHT11 Bridge disabled: python3-msgpack is not installed."
            )
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="DHT11-Bridge"
        )

        self.thread.start()

        print(
            "DHT11 Bridge reader started "
            "(UNO Q D8 via Arduino Bridge)"
        )

    def _rpc_call(self, method):
        if msgpack is None:
            return None

        request_id = next(self.request_ids)

        request = [
            0,
            request_id,
            method,
            []
        ]

        packed = msgpack.packb(
            request,
            use_bin_type=True
        )

        try:
            with self.lock:
                with socket.socket(
                    socket.AF_UNIX,
                    socket.SOCK_STREAM
                ) as client:

                    client.settimeout(2.0)

                    client.connect(
                        self.SOCKET_PATH
                    )

                    client.sendall(
                        packed
                    )

                    unpacker = msgpack.Unpacker(
                        raw=False
                    )

                    while True:
                        chunk = client.recv(4096)

                        if not chunk:
                            return None

                        unpacker.feed(chunk)

                        for response in unpacker:

                            if (
                                isinstance(response, list)
                                and len(response) >= 4
                                and response[0] == 1
                                and response[1] == request_id
                            ):

                                error = response[2]
                                result = response[3]

                                if error is not None:
                                    print(
                                        f"DHT11 RPC error "
                                        f"{method}: {error}"
                                    )
                                    return None

                                try:
                                    return float(result)
                                except Exception:
                                    return None

        except Exception as exc:

            print(
                f"DHT11 Bridge read failed "
                f"({method}): {exc}"
            )

            return None

    def _read_once(self):
        temperature = self._rpc_call(
            "get_dht_temperature"
        )

        humidity = self._rpc_call(
            "get_dht_humidity"
        )

        if (
            temperature is not None
            and
            humidity is not None
        ):

            if -40.0 <= temperature <= 80.0:
                APP.temperature = temperature

            if 0.0 <= humidity <= 100.0:
                APP.humidity = humidity

    def _loop(self):

        # DHT11 should not be polled too aggressively.
        # The MCU updates its cached reading approximately every 2 seconds.
        while self.running:

            try:
                self._read_once()
            except Exception as exc:
                print(
                    "DHT11 Bridge loop error:",
                    exc
                )

            time.sleep(2.0)

    def stop(self):
        self.running = False

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):
            self.thread.join(
                timeout=1.0
            )


DHT = DHTBridgeReader()


# ================================================================
# MODEL MANAGER
# ================================================================

class ModelManager:

    def __init__(self):

        self.models = {}

    def load(self, crop):

        if crop not in MODEL_PATHS:
            raise KeyError(f"Unknown crop: {crop}")

        if tf is None:
            raise RuntimeError(
                "TensorFlow/Keras is not installed. "
                "Install the supported TensorFlow build for the UNO Q."
            )

        if crop in self.models:
            return self.models[crop]

        path = MODEL_PATHS[crop]

        if not path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{path}\n\n"
                "If the repository was cloned with Git LFS, run "
                "`git lfs install` and `git lfs pull`."
            )

        try:
            with path.open("rb") as model_file:
                header = model_file.read(200)
            if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError(
                    f"{path.name} is a Git LFS pointer, not the actual model. "
                    "Run `git lfs install` and `git lfs pull`."
                )
        except OSError as exc:
            raise RuntimeError(f"Cannot read model file: {path}\n{exc}") from exc

        print(
            f"Loading {crop} model:\n{path}"
        )

        model = tf.keras.models.load_model(
            str(path),
            compile=False
        )

        expected = len(
            CLASS_NAMES[crop]
        )

        try:
            output_count = int(
                model.output_shape[-1]
            )
        except Exception:
            output_count = expected

        if output_count != expected:

            raise RuntimeError(
                f"{crop} model has {output_count} "
                f"outputs but CLASS_NAMES has {expected}.\n"
                f"Correct the CLASS_NAMES['{crop}'] "
                f"order/count to match the training report."
            )

        self.models[crop] = model

        print(
            f"{crop}: model ready "
            f"({output_count} classes)"
        )

        return model

    def predict(self, crop, image):

        model = self.load(crop)

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        rgb = cv2.resize(
            rgb,
            (224, 224),
            interpolation=cv2.INTER_AREA
        )

        x = rgb.astype(
            np.float32
        )

        x = np.expand_dims(
            x,
            axis=0
        )

        probabilities = model.predict(
            x,
            verbose=0
        )[0]

        order = np.argsort(
            probabilities
        )[::-1]

        names = CLASS_NAMES[crop]

        top = []

        for index in order[:3]:

            top.append(
                (
                    names[int(index)],
                    float(probabilities[int(index)])
                )
            )

        best_name = top[0][0]
        confidence = top[0][1]

        return (
            best_name,
            confidence,
            top
        )


MODELS = ModelManager()


# ================================================================
# HELPERS
# ================================================================

def translate_disease_name(name):
    lang = APP.language_code
    # Keep the model's exact disease name internally.
    # Translate common crop/disease labels for the selected UI language.
    parts = name.split("_", 1)
    disease_key = parts[1] if len(parts) == 2 else name

    packs = {
        "hi": {
            "BacterialLeafBlight": "बैक्टीरियल लीफ ब्लाइट",
            "BacterialLeafStreak": "बैक्टीरियल लीफ स्ट्रीक",
            "BacterialPanicleBlight": "बैक्टीरियल पैनिकल ब्लाइट",
            "Blast": "ब्लास्ट",
            "BrownSpot": "ब्राउन स्पॉट",
            "Healthy": "स्वस्थ",
            "Tungro": "टुंग्रो",
            "BacterialBlight": "बैक्टीरियल ब्लाइट",
            "Leaf_Hopper_Jassids": "लीफ हॉपर / जैसिड्स",
            "Leaf_Variegation": "लीफ वेरिएगेशन",
            "LeafCurlVirus": "लीफ कर्ल वायरस",
            "LeafRedding": "लीफ रेडनिंग",
            "fussarium_wilt": "फ्यूजेरियम विल्ट",
            "Common_Rust": "कॉमन रस्ट",
            "Gray_Leaf_Spot": "ग्रे लीफ स्पॉट",
            "Leaf_Blight": "लीफ ब्लाइट",
            "RedRot": "रेड रॉट",
            "Rust": "रस्ट",
            "YellowLeaf": "येलो लीफ",
            "Brown_Rust": "ब्राउन रस्ट",
            "Fusarium_Head_Blight": "फ्यूजेरियम हेड ब्लाइट",
            "Septoria": "सेप्टोरिया",
        },
        "te": {
            "BacterialLeafBlight": "బ్యాక్టీరియల్ లీఫ్ బ్లైట్",
            "BacterialLeafStreak": "బ్యాక్టీరియల్ లీఫ్ స్ట్రీక్",
            "BacterialPanicleBlight": "బ్యాక్టీరియల్ పానికల్ బ్లైట్",
            "Blast": "బ్లాస్ట్",
            "BrownSpot": "బ్రౌన్ స్పాట్",
            "Healthy": "ఆరోగ్యంగా ఉంది",
            "Tungro": "టంగ్రో",
            "BacterialBlight": "బ్యాక్టీరియల్ బ్లైట్",
            "Leaf_Hopper_Jassids": "లీఫ్ హాపర్ / జాసిడ్స్",
            "Leaf_Variegation": "లీఫ్ వెరిగేషన్",
            "LeafCurlVirus": "లీఫ్ కర్ల్ వైరస్",
            "LeafRedding": "లీఫ్ రెడ్డెనింగ్",
            "fussarium_wilt": "ఫ్యూజేరియం విల్ట్",
            "Common_Rust": "కామన్ రస్ట్",
            "Gray_Leaf_Spot": "గ్రే లీఫ్ స్పాట్",
            "Leaf_Blight": "లీఫ్ బ్లైట్",
            "RedRot": "రెడ్ రాట్",
            "Rust": "రస్ట్",
            "YellowLeaf": "ఎల్లో లీఫ్",
            "Brown_Rust": "బ్రౌన్ రస్ట్",
            "Fusarium_Head_Blight": "ఫ్యూజేరియం హెడ్ బ్లైట్",
            "Septoria": "సెప్టోరియా",
        },
        "ta": {
            "BacterialLeafBlight": "பாக்டீரியல் இலை கருகல்",
            "BacterialLeafStreak": "பாக்டீரியல் இலை கோடு",
            "BacterialPanicleBlight": "பாக்டீரியல் கதிர் கருகல்",
            "Blast": "பிளாஸ்ட்",
            "BrownSpot": "பிரவுன் ஸ்பாட்",
            "Healthy": "ஆரோக்கியமானது",
            "Tungro": "டங்ரோ",
            "BacterialBlight": "பாக்டீரியல் பிளைட்",
            "Leaf_Hopper_Jassids": "இலைத் தத்துப்பூச்சி / ஜாசிட்ஸ்",
            "Leaf_Variegation": "இலை நிறமாற்றம்",
            "LeafCurlVirus": "இலை சுருட்டு வைரஸ்",
            "LeafRedding": "இலை சிவத்தல்",
            "fussarium_wilt": "பியூசேரியம் வாடல்",
            "Common_Rust": "காமன் ரஸ்ட்",
            "Gray_Leaf_Spot": "கிரே லீஃப் ஸ்பாட்",
            "Leaf_Blight": "லீஃப் பிளைட்",
            "RedRot": "ரெட் ராட்",
            "Rust": "ரஸ்ட்",
            "YellowLeaf": "மஞ்சள் இலை",
            "Brown_Rust": "பிரவுன் ரஸ்ட்",
            "Fusarium_Head_Blight": "பியூசேரியம் தலை கருகல்",
            "Septoria": "செப்டோரியா",
        },
    }

    # More languages can be expanded safely without touching model labels.
    if lang in packs and disease_key in packs[lang]:
        return packs[lang][disease_key]

    # CropAI is still fully translated at the interface level; if a specific
    # disease name lacks a native-language term, keep the scientific/common
    # English disease name rather than inventing an unsafe translation.
    return pretty_name(name)

def pretty_name(name):

    name = name.replace(
        "_",
        " "
    )

    name = name.replace(
        "fussarium",
        "Fusarium"
    )

    return name



CROP_TRANSLATIONS = {
    "hi": {"Corn":"मक्का","Cotton":"कपास","Paddy":"धान","Sugarcane":"गन्ना","Wheat":"गेहूँ"},
    "te": {"Corn":"మొక్కజొన్న","Cotton":"పత్తి","Paddy":"వరి","Sugarcane":"చెరకు","Wheat":"గోధుమ"},
    "ta": {"Corn":"மக்காச்சோளம்","Cotton":"பருத்தி","Paddy":"நெல்","Sugarcane":"கரும்பு","Wheat":"கோதுமை"},
    "kn": {"Corn":"ಮೆಕ್ಕೆಜೋಳ","Cotton":"ಹತ್ತಿ","Paddy":"ಭತ್ತ","Sugarcane":"ಕಬ್ಬು","Wheat":"ಗೋಧಿ"},
    "ml": {"Corn":"ചോളം","Cotton":"പരുത്തി","Paddy":"നെല്ല്","Sugarcane":"കരിമ്പ്","Wheat":"ഗോതമ്പ്"},
    "mr": {"Corn":"मका","Cotton":"कापूस","Paddy":"भात","Sugarcane":"ऊस","Wheat":"गहू"},
    "bn": {"Corn":"ভুট্টা","Cotton":"তুলা","Paddy":"ধান","Sugarcane":"আখ","Wheat":"গম"},
    "gu": {"Corn":"મકાઈ","Cotton":"કપાસ","Paddy":"ડાંગર","Sugarcane":"શેરડી","Wheat":"ઘઉં"},
    "pa": {"Corn":"ਮੱਕੀ","Cotton":"ਕਪਾਹ","Paddy":"ਝੋਨਾ","Sugarcane":"ਗੰਨਾ","Wheat":"ਕਣਕ"},
    "or": {"Corn":"ମକା","Cotton":"କପା","Paddy":"ଧାନ","Sugarcane":"ଆଖୁ","Wheat":"ଗହମ"},
}

def translated_crop_name(crop):
    base = CROPS[crop]["name"]
    return CROP_TRANSLATIONS.get(
        APP.language_code,
        {}
    ).get(base, base)


def crop_display_name(crop):

    return CROPS[crop]["name"]


def confidence_text(value):

    return f"{value * 100:.1f}%"


def weather_text():

    t = (
        f"{APP.temperature:.1f} °C"
        if APP.temperature is not None
        else "-- °C"
    )

    h = (
        f"{APP.humidity:.1f}%"
        if APP.humidity is not None
        else "-- %"
    )

    return t, h


def get_info(disease):

    return DISEASE_INFO.get(
        disease,
        {
            "cause": "Disease-specific cause information is not available in the local knowledge pack.",
            "management": "Follow local agricultural extension recommendations and monitor the crop.",
            "chemicals": "Use only locally registered products and follow the current product label. Do not use a universal dose.",
        }
    )


# ================================================================
# CROP CARD IMAGE GENERATION
# ================================================================

# ================================================================
# REAL CROP CARD IMAGES
#
# Images are sourced from Wikimedia Commons under the licenses
# recorded below. The app downloads them once, caches them locally,
# and uses the cached image on subsequent launches.
#
# If the UNO Q has no internet on first launch, the program falls
# back to a clean photo-style placeholder. To guarantee offline use,
# run the app once on an internet-connected machine and copy the
# generated crop_cards/ directory to the UNO Q.
# ================================================================

REAL_CROP_SOURCES = {
    "corn": {
        "page": "https://commons.wikimedia.org/wiki/File:Single_Maize_plant.jpg",
        "license": "CC BY 4.0 — Wikimedia Commons, Single Maize plant",
    },
    "cotton": {
        "page": "https://commons.wikimedia.org/wiki/File:Field_of_Cotton_Plants.jpg",
        "license": "CC0 1.0 — Wikimedia Commons, Field of Cotton Plants",
    },
    "paddy": {
        "page": "https://commons.wikimedia.org/wiki/File:Rice_plant.jpg",
        "license": "CC BY 4.0 — Wikimedia Commons, Rice plant",
    },
    "sugarcane": {
        "page": "https://commons.wikimedia.org/wiki/File:Sugarcane_field_in_Queensland,_Australia_3.jpg",
        "license": "CC BY 2.0 — Wikimedia Commons, Sugarcane field in Queensland",
    },
    "wheat": {
        "page": "https://commons.wikimedia.org/wiki/File:Wheat_Field_Crop.jpg",
        "license": "CC BY-SA 4.0 — Wikimedia Commons, Wheat Field Crop",
    },
}


def download_real_crop_image(crop, destination):
    """
    Download the real crop photograph from the Commons page's
    og:image URL. Returns True when successful.
    """
    source = REAL_CROP_SOURCES.get(crop)
    if not source:
        return False

    try:
        headers = {
            "User-Agent":
                "CropAI/1.0 (educational agriculture project)"
        }

        request = Request(
            source["page"],
            headers=headers
        )

        with urlopen(request, timeout=12) as response:
            html = response.read().decode(
                "utf-8",
                errors="ignore"
            )

        # Wikimedia exposes the image used by the page as og:image.
        match = re.search(
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE
        )

        if not match:
            match = re.search(
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
                html,
                re.IGNORECASE
            )

        if not match:
            return False

        image_url = match.group(1)

        with urlopen(
            Request(
                image_url,
                headers=headers
            ),
            timeout=20
        ) as response:

            data = response.read()

        destination.write_bytes(data)

        return destination.exists() and destination.stat().st_size > 1000

    except Exception as error:
        print(
            f"Real crop image download failed for {crop}: {error}"
        )
        return False


def _fit_cover(pil_image, size):
    """Crop/resize a PIL image to exactly fit size."""
    target_w, target_h = size
    image = pil_image.convert("RGB")

    iw, ih = image.size

    if iw <= 0 or ih <= 0:
        return Image.new(
            "RGB",
            size,
            "#16222d"
        )

    scale = max(
        target_w / iw,
        target_h / ih
    )

    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))

    image = image.resize(
        (nw, nh),
        Image.Resampling.LANCZOS
    )

    left = max(
        0,
        (nw - target_w) // 2
    )

    top = max(
        0,
        (nh - target_h) // 2
    )

    return image.crop(
        (
            left,
            top,
            left + target_w,
            top + target_h
        )
    )


def make_crop_card(crop):

    path = (
        CROP_CARD_DIR /
        f"{crop}.png"
    )

    # Already cached.
    if path.exists():
        return path

    raw_path = (
        CROP_CARD_DIR /
        f"{crop}_real_source.jpg"
    )

    # Try to obtain a real photograph.
    if (
        not raw_path.exists()
        or raw_path.stat().st_size < 1000
    ):

        download_real_crop_image(
            crop,
            raw_path
        )

    # ------------------------------------------------------------
    # Real photograph available
    # ------------------------------------------------------------

    if (
        raw_path.exists()
        and raw_path.stat().st_size >= 1000
        and Image is not None
    ):

        try:

            real = Image.open(
                raw_path
            )

            # Card image region.
            card_w = 360
            card_h = 220

            card = _fit_cover(
                real,
                (card_w, card_h)
            ).copy()

            draw = ImageDraw.Draw(
                card,
                "RGBA"
            )

            # Dark transparent top strip.
            draw.rectangle(
                (0, 0, card_w, 48),
                fill=(2, 7, 12, 175)
            )

            # Dark transparent bottom strip.
            draw.rectangle(
                (0, card_h - 55, card_w, card_h),
                fill=(2, 7, 12, 190)
            )

            # Futuristic border.
            draw.rounded_rectangle(
                (3, 3, card_w - 3, card_h - 3),
                radius=18,
                outline=(0, 230, 184, 255),
                width=3
            )

            draw.text(
                (18, 15),
                CROPS[crop]["name"],
                fill=(245, 250, 252, 255)
            )

            draw.text(
                (18, card_h - 40),
                "REAL CROP IMAGE",
                fill=(0, 230, 184, 255)
            )

            card.save(
                path,
                "PNG",
                optimize=True
            )

            return path

        except Exception as error:

            print(
                f"Could not build photo card for {crop}: {error}"
            )

    # ------------------------------------------------------------
    # Offline fallback
    # ------------------------------------------------------------

    img = Image.new(
        "RGB",
        (360, 220),
        (20, 29, 38)
    )

    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        (5, 5, 355, 215),
        radius=18,
        outline=(0, 230, 184),
        width=3
    )

    # Keep the former stylised drawing only as an offline fallback.
    cx = 180

    if crop == "corn":

        draw.line(
            (180, 190, 180, 55),
            fill=(70, 160, 80),
            width=9
        )

        for side in (-1, 1):

            draw.polygon(
                [
                    (180, 145),
                    (180 + side * 100, 105),
                    (180 + side * 35, 155),
                ],
                fill=(80, 170, 85)
            )

        draw.ellipse(
            (155, 35, 205, 100),
            fill=(235, 190, 55)
        )

    elif crop == "cotton":

        draw.line(
            (180, 190, 180, 85),
            fill=(70, 160, 80),
            width=9
        )

        for x, y in [
            (135, 100),
            (225, 105),
            (155, 65),
            (205, 65)
        ]:

            draw.ellipse(
                (x-28, y-28, x+28, y+28),
                fill=(240, 240, 235)
            )

    elif crop == "paddy":

        for x in range(
            110,
            251,
            22
        ):

            draw.line(
                (x, 190, x+10, 70),
                fill=(80, 175, 85),
                width=5
            )

    elif crop == "sugarcane":

        for x in range(
            110,
            251,
            30
        ):

            draw.line(
                (x, 190, x-20, 55),
                fill=(65, 165, 80),
                width=13
            )

    elif crop == "wheat":

        for x in range(
            120,
            241,
            25
        ):

            draw.line(
                (x, 190, x+10, 60),
                fill=(90, 160, 70),
                width=5
            )

    draw.text(
        (20, 15),
        CROPS[crop]["name"],
        fill=(245, 245, 245)
    )

    draw.text(
        (20, 190),
        "PHOTO NOT CACHED",
        fill=(255, 200, 80)
    )

    img.save(
        path,
        "PNG"
    )

    return path


def write_crop_image_credits():
    """
    Store source/license information locally so the deployed
    application contains attribution for the real photographs.
    """
    credits_path = ROOT / "crop_image_credits.txt"

    try:

        with open(
            credits_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "CropAI crop-card image sources\n"
                "================================\n\n"
            )

            for crop, info in REAL_CROP_SOURCES.items():

                f.write(
                    f"{CROPS[crop]['name']}\n"
                )

                f.write(
                    f"Source: {info['page']}\n"
                )

                f.write(
                    f"License: {info['license']}\n\n"
                )

    except Exception as error:
        print(
            "Could not write crop image credits:",
            error
        )


write_crop_image_credits()


# ================================================================
# TKINTER APPLICATION
# ================================================================

class CropAIApp:

    def __init__(self):

        if tk is None:
            raise RuntimeError(
                "Tkinter is not installed."
            )

        self.root = tk.Tk()

        self.root.title(
            "CropAI — One AI Farm Assistant"
        )

        self.root.geometry(
            "1280x800"
        )

        self.root.configure(
            bg="#10161d"
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.quit_app
        )

        # Keyboard controls (mouse buttons remain available everywhere)
        self.root.bind("<Escape>", lambda e: self.quit_app())
        self.root.bind("<F1>", lambda e: self.show_profile())
        self.root.bind("<F2>", lambda e: self.show_crop_selection())
        self.root.bind("<space>", self._keyboard_capture)
        self.root.bind("c", self._keyboard_capture)
        self.root.bind("C", self._keyboard_capture)
        self.root.bind("r", self._keyboard_recapture)
        self.root.bind("R", self._keyboard_recapture)
        self.root.bind("u", self._keyboard_upload)
        self.root.bind("U", self._keyboard_upload)
        self.root.bind("s", self._keyboard_save)
        self.root.bind("S", self._keyboard_save)
        self.root.bind("b", self._keyboard_back)
        self.root.bind("B", self._keyboard_back)

        self.camera = None
        self.camera_running = False
        self.camera_frame = None

        self.photo = None

        self.setup_style()

        self.show_profile()



    def _entry_has_focus(self):
        try:
            widget = self.root.focus_get()
            return isinstance(widget, (tk.Entry, tk.Text))
        except Exception:
            return False

    def _keyboard_capture(self, event=None):
        if self._entry_has_focus():
            return
        if hasattr(self, "capture"):
            try:
                if self.camera_running:
                    self.capture()
                    return "break"
            except Exception:
                pass

    def _keyboard_recapture(self, event=None):
        if self._entry_has_focus():
            return
        if hasattr(self, "recapture"):
            try:
                self.recapture()
                return "break"
            except Exception:
                pass

    def _keyboard_upload(self, event=None):
        if self._entry_has_focus():
            return
        if hasattr(self, "upload_image"):
            try:
                self.upload_image()
                return "break"
            except Exception:
                pass

    def _keyboard_save(self, event=None):
        if self._entry_has_focus():
            return
        if hasattr(self, "save_report"):
            try:
                self.save_report()
                return "break"
            except Exception:
                pass

    def _keyboard_back(self, event=None):
        if self._entry_has_focus():
            return
        if hasattr(self, "back_to_crops"):
            try:
                self.back_to_crops()
                return "break"
            except Exception:
                pass

    # ============================================================
    # STYLE
    # ============================================================

    def setup_style(self):

        self.bg = "#070b12"
        self.panel = "#0d1622"
        self.panel2 = "#142233"
        self.text = "#e8f7ff"
        self.muted = "#7f9aaa"
        self.green = "#00e6b8"
        self.orange = "#ffc857"
        self.red = "#ff5577"

    def clear(self):

        for widget in self.root.winfo_children():
            widget.destroy()


    def title(self, parent, text, size=28):

        return tk.Label(
            parent,
            text=text,
            bg=self.bg,
            fg=self.text,
            font=("DejaVu Sans", size, "bold")
        )


    def button(
        self,
        parent,
        text,
        command,
        width=18
    ):

        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            height=2,
            bg=self.green,
            fg="#07110e",
            activebackground="#65ddb8",
            relief="flat",
            font=("DejaVu Sans", 12, "bold"),
            cursor="hand2",
            activeforeground="#061018",
            bd=0
        )


    # ============================================================
    # STARTUP PROFILE
    # ============================================================

    def show_profile(self):

        self.stop_camera()
        self.clear()

        frame = tk.Frame(
            self.root,
            bg=self.bg
        )

        frame.pack(
            fill="both",
            expand=True,
            padx=80,
            pady=50
        )

        self.title(
            frame,
            tr("welcome"),
            32
        ).pack(
            pady=(20, 8)
        )

        tk.Label(
            frame,
            text=tr("profile"),
            bg=self.bg,
            fg=self.muted,
            font=("DejaVu Sans", 15)
        ).pack(
            pady=(0, 10)
        )

        tk.Label(
            frame,
            text="Type normally • use the button or Ctrl+Enter to continue",
            bg=self.bg,
            fg=self.green,
            font=("DejaVu Sans", 10)
        ).pack(
            pady=(0, 20)
        )

        form = tk.Frame(
            frame,
            bg=self.bg
        )

        form.pack()

        tk.Label(
            form,
            text=tr("name"),
            bg=self.bg,
            fg=self.text,
            font=("DejaVu Sans", 13)
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=10
        )

        self.name_entry = tk.Entry(
            form,
            width=35,
            bg=self.panel2,
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("DejaVu Sans", 13)
        )

        self.name_entry.grid(
            row=0,
            column=1,
            padx=10,
            pady=10
        )

        self.name_entry.bind(
            "<Control-Return>",
            lambda event: self.save_profile()
        )

        tk.Label(
            form,
            text=tr("place"),
            bg=self.bg,
            fg=self.text,
            font=("DejaVu Sans", 13)
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=10,
            pady=10
        )

        self.place_entry = tk.Entry(
            form,
            width=35,
            bg=self.panel2,
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("DejaVu Sans", 13)
        )

        self.place_entry.grid(
            row=1,
            column=1,
            padx=10,
            pady=10
        )

        # Keep keyboard focus in the entry while typing. Enter is deliberately
        # NOT bound globally because that would submit the profile accidentally.
        self.place_entry.bind("<Return>", lambda event: "break")
        self.place_entry.bind(
            "<Control-Return>",
            lambda event: self.save_profile()
        )

        tk.Label(
            form,
            text=tr("language"),
            bg=self.bg,
            fg=self.text,
            font=("DejaVu Sans", 13)
        ).grid(
            row=2,
            column=0,
            sticky="w",
            padx=10,
            pady=10
        )

        self.language_var = tk.StringVar(
            value="English"
        )

        tk.OptionMenu(
            form,
            self.language_var,
            *LANGUAGES.keys()
        ).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=10,
            pady=10
        )

        self.button(
            frame,
            tr("continue") + "  [CTRL+ENTER]",
            self.save_profile,
            width=22
        ).pack(
            pady=35
        )

        tk.Label(
            frame,
            text="CropAI • UNO Q • AI-assisted crop disease screening",
            bg=self.bg,
            fg=self.muted,
            font=("DejaVu Sans", 10)
        ).pack(
            side="bottom",
            pady=20
        )


    def save_profile(self):

        APP.name = (
            self.name_entry.get().strip()
            or "Farmer"
        )

        APP.place = (
            self.place_entry.get().strip()
            or "Not specified"
        )

        APP.language = (
            self.language_var.get()
        )

        APP.language_code = LANGUAGES.get(
            APP.language,
            "en"
        )

        self.show_crop_selection()


    # ============================================================
    # CROP SELECTION
    # ============================================================

    def show_crop_selection(self):

        self.clear()

        self.title(
            self.root,
            tr("select_crop"),
            28
        ).pack(
            pady=25
        )

        cards = tk.Frame(
            self.root,
            bg=self.bg
        )

        cards.pack(
            fill="both",
            expand=True,
            padx=30,
            pady=10
        )

        self.crop_images = []

        for col, crop in enumerate(CROPS):

            card = tk.Frame(
                cards,
                bg=self.panel,
                bd=0
            )

            card.grid(
                row=0,
                column=col,
                padx=8,
                pady=10,
                sticky="nsew"
            )

            cards.grid_columnconfigure(
                col,
                weight=1
            )

            path = make_crop_card(
                crop
            )

            try:

                img = Image.open(
                    path
                ).resize(
                    (220, 135)
                )

                photo = ImageTk.PhotoImage(
                    img
                )

                self.crop_images.append(
                    photo
                )

                tk.Label(
                    card,
                    image=photo,
                    bg=self.panel
                ).pack(
                    padx=10,
                    pady=12
                )

            except Exception:

                tk.Label(
                    card,
                    text=CROPS[crop]["emoji"],
                    bg=self.panel,
                    fg=self.text,
                    font=("DejaVu Sans", 45)
                ).pack(
                    pady=35
                )

            tk.Label(
                card,
                text=translated_crop_name(crop),
                bg=self.panel,
                fg=self.text,
                font=("DejaVu Sans", 17, "bold")
            ).pack()

            tk.Label(
                card,
                text=CROPS[crop]["subtitle"],
                bg=self.panel,
                fg=self.muted,
                wraplength=190,
                font=("DejaVu Sans", 10)
            ).pack(
                pady=8
            )

            self.button(
                card,
                tr("start"),
                lambda c=crop: self.start_crop(c),
                width=14
            ).pack(
                pady=(10, 18)
            )

        bottom = tk.Frame(
            self.root,
            bg=self.bg
        )

        bottom.pack(
            pady=15
        )

        self.button(
            bottom,
            tr("quit"),
            self.quit_app,
            width=12
        ).pack(
            side="left",
            padx=8
        )


    # ============================================================
    # START SELECTED CROP
    # ============================================================

    def start_crop(self, crop):

        APP.crop = crop

        try:

            self.status_popup(
                tr("loading")
            )

            MODELS.load(
                crop
            )

            self.close_popup()

            self.show_detector()

        except Exception as e:

            self.close_popup()

            if messagebox:

                messagebox.showerror(
                    "CropAI",
                    str(e)
                )

            else:

                print(
                    "MODEL ERROR:",
                    e
                )


    def status_popup(self, text):

        self.popup = tk.Toplevel(
            self.root
        )

        self.popup.title(
            "CropAI"
        )

        self.popup.configure(
            bg=self.bg
        )

        tk.Label(
            self.popup,
            text=text,
            bg=self.bg,
            fg=self.text,
            font=("DejaVu Sans", 14)
        ).pack(
            padx=40,
            pady=35
        )

        self.popup.update_idletasks()


    def close_popup(self):

        try:
            self.popup.destroy()
        except Exception:
            pass


    # ============================================================
    # DETECTOR SCREEN
    # ============================================================

    def show_detector(self):

        self.clear()

        # --------------------------------------------------------
        # FUTURISTIC HEADER
        # --------------------------------------------------------
        header = tk.Frame(
            self.root,
            bg="#050a11",
            height=74
        )
        header.pack(fill="x")
        header.pack_propagate(False)

        title_frame = tk.Frame(
            header,
            bg="#050a11"
        )
        title_frame.pack(side="left", padx=20, pady=8)

        tk.Label(
            title_frame,
            text="CROPAI  //  ONE AI",
            bg="#050a11",
            fg=self.green,
            font=("DejaVu Sans Mono", 17, "bold")
        ).pack(anchor="w")

        tk.Label(
            title_frame,
            text=(
                f"{CROPS[APP.crop]['name']}  •  AI DISEASE SCREENING"
            ),
            bg="#050a11",
            fg=self.muted,
            font=("DejaVu Sans", 9)
        ).pack(anchor="w")

        weather_box = tk.Frame(
            header,
            bg="#0a1420",
            highlightbackground="#193244",
            highlightthickness=1
        )
        weather_box.pack(
            side="right",
            padx=18,
            pady=10
        )

        self.weather_label = tk.Label(
            weather_box,
            text="Temperature: -- °C    Humidity: -- %",
            bg="#0a1420",
            fg=self.green,
            font=("DejaVu Sans Mono", 10, "bold")
        )
        self.weather_label.pack(
            padx=15,
            pady=8
        )

        # --------------------------------------------------------
        # MAIN AREA
        # --------------------------------------------------------
        main = tk.Frame(
            self.root,
            bg=self.bg
        )
        main.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=(10, 8)
        )

        # --------------------------------------------------------
        # CAMERA PANEL
        # --------------------------------------------------------
        camera_panel = tk.Frame(
            main,
            bg="#05090f",
            highlightbackground="#173445",
            highlightthickness=1
        )
        camera_panel.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 10)
        )

        camera_top = tk.Frame(
            camera_panel,
            bg="#09121c",
            height=44
        )
        camera_top.pack(fill="x")
        camera_top.pack_propagate(False)

        tk.Label(
            camera_top,
            text="LIVE INPUT",
            bg="#09121c",
            fg=self.green,
            font=("DejaVu Sans Mono", 10, "bold")
        ).pack(side="left", padx=14)

        tk.Label(
            camera_top,
            text="WEBCAM / IMAGE UPLOAD",
            bg="#09121c",
            fg=self.muted,
            font=("DejaVu Sans Mono", 8)
        ).pack(side="right", padx=14)

        self.video_label = tk.Label(
            camera_panel,
            bg="#02050a",
            text="",
            bd=0
        )
        self.video_label.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        camera_hint = tk.Label(
            camera_panel,
            text=(
                "Place ONE leaf inside the box  •  "
                "Good lighting  •  Keep the leaf in focus"
            ),
            bg="#07101a",
            fg=self.muted,
            font=("DejaVu Sans", 9)
        )
        camera_hint.pack(
            fill="x",
            padx=10,
            pady=(0, 10)
        )

        # --------------------------------------------------------
        # INFORMATION PANEL
        # --------------------------------------------------------
        info_panel = tk.Frame(
            main,
            bg="#0d1622",
            width=430,
            highlightbackground="#173445",
            highlightthickness=1
        )
        info_panel.pack(
            side="right",
            fill="y"
        )
        info_panel.pack_propagate(False)

        # Crop/model badge
        crop_head = tk.Frame(
            info_panel,
            bg="#101c2a",
            height=82
        )
        crop_head.pack(fill="x")
        crop_head.pack_propagate(False)

        tk.Label(
            crop_head,
            text=translated_crop_name(APP.crop).upper(),
            bg="#101c2a",
            fg=self.green,
            font=("DejaVu Sans", 22, "bold")
        ).pack(
            anchor="w",
            padx=18,
            pady=(10, 0)
        )

        tk.Label(
            crop_head,
            text="CROP-SPECIFIC .KERAS MODEL ACTIVE",
            bg="#101c2a",
            fg=self.muted,
            font=("DejaVu Sans Mono", 8)
        ).pack(
            anchor="w",
            padx=18
        )

        # Result card
        result_card = tk.Frame(
            info_panel,
            bg="#111f2d",
            highlightbackground="#214153",
            highlightthickness=1
        )
        result_card.pack(
            fill="x",
            padx=12,
            pady=(12, 8)
        )

        tk.Label(
            result_card,
            text=tr("result"),
            bg="#111f2d",
            fg=self.muted,
            font=("DejaVu Sans Mono", 9, "bold")
        ).pack(
            anchor="w",
            padx=14,
            pady=(10, 0)
        )

        self.result_label = tk.Label(
            result_card,
            text=(
                "READY\n"
                "Capture or upload a leaf image"
            ),
            bg="#111f2d",
            fg=self.text,
            justify="left",
            anchor="w",
            wraplength=375,
            font=("DejaVu Sans", 14, "bold")
        )
        self.result_label.pack(
            fill="x",
            padx=14,
            pady=(4, 10)
        )

        # Top prediction card
        top_card = tk.Frame(
            info_panel,
            bg="#0f1b28"
        )
        top_card.pack(
            fill="x",
            padx=12,
            pady=4
        )

        self.top_label = tk.Label(
            top_card,
            text=(
                "TOP 3 PREDICTIONS\n"
                "────────────────────────\n"
                "No analysis yet"
            ),
            bg="#0f1b28",
            fg=self.text,
            justify="left",
            anchor="w",
            wraplength=375,
            font=("DejaVu Sans Mono", 9)
        )
        self.top_label.pack(
            fill="x",
            padx=14,
            pady=10
        )

        # Scrollable detailed information area
        details_frame = tk.Frame(
            info_panel,
            bg="#0d1622"
        )
        details_frame.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=6
        )

        self.detail_text = scrolledtext.ScrolledText(
            details_frame,
            wrap="word",
            bg="#08111b",
            fg="#d9e7ef",
            insertbackground=self.text,
            selectbackground="#17475a",
            relief="flat",
            borderwidth=0,
            font=("DejaVu Sans", 9),
            padx=12,
            pady=10
        )
        self.detail_text.pack(
            fill="both",
            expand=True
        )

        self.detail_text.insert(
            "1.0",
            "INFORMATION\n"
            "────────────────────────\n"
            "Disease symptoms, cause, risk conditions, "
            "management, chemical options, monitoring plan "
            "and DHT11 readings will appear here after analysis.\n\n"
            "Use CAPTURE or UPLOAD to start."
        )
        self.detail_text.configure(
            state="disabled"
        )

        # --------------------------------------------------------
        # ACTION BAR
        # --------------------------------------------------------
        action_bar = tk.Frame(
            info_panel,
            bg="#07101a"
        )
        action_bar.pack(
            fill="x",
            padx=12,
            pady=(4, 12)
        )

        # First row
        capture_btn = self.button(
            action_bar,
            "CAPTURE  [C]",
            self.capture,
            width=14
        )
        capture_btn.grid(
            row=0, column=0,
            padx=4, pady=4
        )

        upload_btn = self.button(
            action_bar,
            "UPLOAD  [U]",
            self.upload_image,
            width=14
        )
        upload_btn.grid(
            row=0, column=1,
            padx=4, pady=4
        )

        # Second row
        recapture_btn = self.button(
            action_bar,
            "RECAPTURE  [R]",
            self.recapture,
            width=14
        )
        recapture_btn.configure(
            bg=self.orange,
            activebackground="#ffd979"
        )
        recapture_btn.grid(
            row=1, column=0,
            padx=4, pady=4
        )

        save_btn = self.button(
            action_bar,
            "SAVE REPORT  [S]",
            self.save_report,
            width=14
        )
        save_btn.grid(
            row=1, column=1,
            padx=4, pady=4
        )

        # Third row
        back_btn = tk.Button(
            action_bar,
            text="BACK  [B]",
            command=self.back_to_crops,
            width=14,
            height=2,
            bg=self.panel2,
            fg=self.text,
            activebackground="#20384a",
            activeforeground=self.text,
            relief="flat",
            font=("DejaVu Sans", 10, "bold"),
            cursor="hand2",
            bd=0
        )
        back_btn.grid(
            row=2, column=0,
            padx=4, pady=4
        )

        quit_btn = tk.Button(
            action_bar,
            text="EXIT  [ESC]",
            command=self.quit_app,
            width=14,
            height=2,
            bg="#3a1520",
            fg="#ffb9c7",
            activebackground="#55202e",
            activeforeground="#ffffff",
            relief="flat",
            font=("DejaVu Sans", 10, "bold"),
            cursor="hand2",
            bd=0
        )
        quit_btn.grid(
            row=2, column=1,
            padx=4, pady=4
        )

        # Full-width shortcut guide
        shortcut_bar = tk.Frame(
            self.root,
            bg="#03070c",
            height=34
        )
        shortcut_bar.pack(
            fill="x",
            side="bottom"
        )

        tk.Label(
            shortcut_bar,
            text=(
                "SPACE / C  Capture     "
                "R  Recapture     "
                "U  Upload     "
                "S  Save PDF/JPG/PNG     "
                "B  Back     "
                "ESC  Exit"
            ),
            bg="#03070c",
            fg=self.green,
            font=("DejaVu Sans Mono", 9, "bold")
        ).pack(pady=7)

        self.start_camera()
        self.update_weather()

    # ============================================================
    # CAMERA
    # ============================================================

    def start_camera(self):

        self.stop_camera()

        # External webcam is normally 0/1 depending on UNO Q.
        camera_index = int(
            os.environ.get(
                "CROPAI_CAMERA",
                "0"
            )
        )

        self.camera = cv2.VideoCapture(
            camera_index,
            cv2.CAP_V4L2
        )

        if not self.camera.isOpened():

            self.camera = cv2.VideoCapture(
                camera_index
            )

        if not self.camera.isOpened():

            if messagebox:

                messagebox.showerror(
                    "CropAI",
                    "External webcam could not be opened.\n\n"
                    "Try:\n"
                    "CROPAI_CAMERA=1 python3 cropai_one_ai.py"
                )

            return

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        self.camera_running = True

        self.update_camera()


    def update_camera(self):

        if not self.camera_running:
            return

        ok, frame = self.camera.read()

        if ok:

            self.camera_frame = frame.copy()

            display = frame.copy()

            h, w = display.shape[:2]

            # Detection box.
            box_w = int(w * 0.68)
            box_h = int(h * 0.78)

            x1 = (w - box_w) // 2
            y1 = (h - box_h) // 2

            x2 = x1 + box_w
            y2 = y1 + box_h

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 220, 160),
                3
            )

            cv2.rectangle(
                display,
                (0, 0),
                (display.shape[1], 82),
                (3, 9, 16),
                -1
            )

            cv2.putText(
                display,
                "CROPAI  //  LIVE LEAF SCAN",
                (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 230, 184),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                "PLACE ONE LEAF INSIDE THE BOX  •  C = CAPTURE  •  U = UPLOAD  •  R = RECAPTURE",
                (24, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (200, 220, 230),
                1,
                cv2.LINE_AA
            )

            rgb = cv2.cvtColor(
                display,
                cv2.COLOR_BGR2RGB
            )

            img = Image.fromarray(
                rgb
            )

            # Fit to display area.
            max_w = 820
            max_h = 680

            ratio = min(
                max_w / img.width,
                max_h / img.height
            )

            new_size = (
                int(img.width * ratio),
                int(img.height * ratio)
            )

            img = img.resize(
                new_size,
                Image.Resampling.LANCZOS
            )

            self.photo = ImageTk.PhotoImage(
                img
            )

            self.video_label.configure(
                image=self.photo
            )

        self.root.after(
            20,
            self.update_camera
        )


    def stop_camera(self):

        self.camera_running = False

        if self.camera is not None:

            try:
                self.camera.release()
            except Exception:
                pass

            self.camera = None


    def recapture(self):

        self.current_image = None

        self.result_label.configure(
            text="READY\nCapture or upload a leaf image",
            fg=self.text
        )

        self.top_label.configure(
            text=""
        )

        if hasattr(self, "detail_text"):
            self.detail_text.configure(
                state="normal"
            )
            self.detail_text.delete(
                "1.0",
                "end"
            )
            self.detail_text.insert(
                "1.0",
                "INFORMATION\n"
                "────────────────────────\n"
                "Ready for a new capture."
            )
            self.detail_text.configure(
                state="disabled"
            )

        if not self.camera_running:
            self.start_camera()


    def build_information_text(self, disease):
        info = translated_info(get_info(disease))
        temp, humidity = weather_text()
        d = DIAG_TRANSLATIONS.get(
            APP.language_code,
            DIAG_TRANSLATIONS["en"]
        )

        detected_name = translate_disease_name(disease)

        return (
            f"{d['disease_information']}\n"
            f"────────────────────────────────\n"
            f"{d['detected']}: {detected_name}\n"
            f"{d['confidence']}: {confidence_text(APP.confidence)}\n\n"

            f"{d['symptoms']}\n"
            f"• {info.get('symptoms', 'Not available.')}\n\n"

            f"{d['cause']}\n"
            f"• {info.get('cause', 'Not available.')}\n\n"

            f"{d['conditions']}\n"
            f"• {info.get('conditions', 'Not available.')}\n\n"

            f"{d['management']}\n"
            f"• {info.get('management', 'Not available.')}\n\n"

            f"{d['chemicals']}\n"
            f"• {info.get('chemicals', 'Use only locally registered products.')}\n\n"

            f"{d['spray']}\n"
            f"• {d['no_universal_dose']} "
            f"{d['use_label']}\n\n"

            f"{d['monitoring']}\n"
            f"• {info.get('monitoring', 'Continue regular field scouting.')}\n\n"

            f"{d['environment']}\n"
            f"• {d['temperature']}: {temp}\n"
            f"• {d['humidity']}: {humidity}\n"
            f"• {d['context']}\n\n"

            f"{d['confirmation']}\n"
            f"• {d['recapture_hint']}\n\n"

            f"{d['field_note']}\n"
            f"• {d['use_label']}"
        )

    # ============================================================
    # PREDICTION
    # ============================================================

    def run_prediction(
        self,
        image,
        source
    ):

        if image is None:
            return

        APP.current_image = image.copy()
        APP.current_source = source

        try:

            (
                disease,
                confidence,
                top
            ) = MODELS.predict(
                APP.crop,
                image
            )

            APP.result_name = disease
            APP.confidence = confidence
            APP.top_predictions = top

            # Conservative uncertainty threshold.
            if confidence < 0.60:

                result_text = (
                    f"{tr('result')}: "
                    f"{tr('uncertain')}\n\n"
                    f"{tr('confidence')}: "
                    f"{confidence_text(confidence)}"
                )

                result_color = self.orange

            else:

                result_text = (
                    f"{tr('result')}: "
                    f"{translate_disease_name(disease)}\n\n"
                    f"{tr('confidence')}: "
                    f"{confidence_text(confidence)}"
                )

                result_color = self.green

            self.result_label.configure(
                text=result_text,
                fg=result_color
            )

            top_text = (
                tr("top3")
                + "\n\n"
            )

            for name, score in top:

                top_text += (
                    f"{pretty_name(name)}"
                    f"    {confidence_text(score)}\n"
                )

            self.top_label.configure(
                text=top_text
            )

            details = self.build_information_text(
                disease
            )

            self.detail_text.configure(
                state="normal"
            )
            self.detail_text.delete(
                "1.0",
                "end"
            )
            self.detail_text.insert(
                "1.0",
                details
            )
            self.detail_text.configure(
                state="disabled"
            )

        except Exception as e:

            self.result_label.configure(
                text=f"Prediction error:\n{e}",
                fg=self.red
            )


    # ============================================================
    # CAPTURE
    # ============================================================

    def capture(self):

        if self.camera_frame is None:
            return

        image = self.camera_frame.copy()

        self.run_prediction(
            image,
            "WEBCAM"
        )


    # ============================================================
    # UPLOAD
    # ============================================================

    def upload_image(self):

        if filedialog is None:
            return

        path = filedialog.askopenfilename(

            title=tr("upload"),

            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.bmp *.webp"
                ),
                (
                    "All files",
                    "*.*"
                )
            ]
        )

        if not path:
            return

        image = cv2.imread(
            path
        )

        if image is None:

            if messagebox:

                messagebox.showerror(
                    "CropAI",
                    "Could not read the selected image."
                )

            return

        self.current_image = image

        # Show uploaded image.
        self.show_uploaded_image(
            image
        )

        self.run_prediction(
            image,
            "IMAGE UPLOAD"
        )


    def show_uploaded_image(self, image):

        display = image.copy()

        h, w = display.shape[:2]

        cv2.rectangle(
            display,
            (int(w * 0.10), int(h * 0.10)),
            (int(w * 0.90), int(h * 0.90)),
            (0, 220, 160),
            3
        )

        rgb = cv2.cvtColor(
            display,
            cv2.COLOR_BGR2RGB
        )

        img = Image.fromarray(
            rgb
        )

        max_w = 820
        max_h = 680

        ratio = min(
            max_w / img.width,
            max_h / img.height
        )

        img = img.resize(
            (
                int(img.width * ratio),
                int(img.height * ratio)
            ),
            Image.Resampling.LANCZOS
        )

        self.photo = ImageTk.PhotoImage(
            img
        )

        self.video_label.configure(
            image=self.photo
        )


    # ============================================================
    # SAVE IMAGE + PDF
    # ============================================================

    def save_report(self):

        if APP.current_image is None:

            if messagebox:

                messagebox.showinfo(
                    "CropAI",
                    "Capture or upload an image first."
                )

            return

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        safe_crop = APP.crop

        base = (
            OUTPUT_DIR /
            f"{safe_crop}_{timestamp}"
        )

        jpg_path = Path(
            str(base) + ".jpg"
        )

        png_path = Path(
            str(base) + ".png"
        )

        pdf_path = Path(
            str(base) + ".pdf"
        )

        cv2.imwrite(
            str(jpg_path),
            APP.current_image
        )

        cv2.imwrite(
            str(png_path),
            APP.current_image
        )

        self.make_pdf(
            pdf_path,
            jpg_path
        )

        if messagebox:

            messagebox.showinfo(
                "CropAI",
                f"{tr('saved')}:\n{pdf_path}"
            )

        print(
            "\nSaved:"
        )

        print(
            jpg_path
        )

        print(
            png_path
        )

        print(
            pdf_path
        )


    def make_pdf(
        self,
        pdf_path,
        image_path
    ):

        if canvas is None:

            raise RuntimeError(
                "reportlab is not installed."
            )

        c = canvas.Canvas(
            str(pdf_path),
            pagesize=A4
        )

        page_w, page_h = A4

        margin = 40

        y = page_h - margin

        c.setFont(
            "Helvetica-Bold",
            20
        )

        c.drawString(
            margin,
            y,
            "CropAI — Crop Disease Analysis"
        )

        y -= 30

        c.setFont(
            "Helvetica",
            10
        )

        c.drawString(
            margin,
            y,
            f"Farmer: {APP.name}"
        )

        y -= 15

        c.drawString(
            margin,
            y,
            f"Place: {APP.place}"
        )

        y -= 15

        c.drawString(
            margin,
            y,
            f"Language: {APP.language}"
        )

        y -= 15

        c.drawString(
            margin,
            y,
            f"Crop: {crop_display_name(APP.crop)}"
        )

        y -= 15

        c.drawString(
            margin,
            y,
            f"Source: {APP.current_source}"
        )

        y -= 15

        c.drawString(
            margin,
            y,
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        y -= 25

        c.setFont(
            "Helvetica-Bold",
            13
        )

        c.drawString(
            margin,
            y,
            f"Result: {pretty_name(APP.result_name)}"
        )

        y -= 18

        c.setFont(
            "Helvetica",
            11
        )

        c.drawString(
            margin,
            y,
            f"Confidence: {confidence_text(APP.confidence)}"
        )

        y -= 25

        temp, humidity = weather_text()

        c.drawString(
            margin,
            y,
            f"Temperature: {temp}"
        )

        y -= 15

        c.drawString(
            margin,
            y,
            f"Humidity: {humidity}"
        )

        y -= 25

        # Image.
        max_image_w = page_w - 2 * margin
        max_image_h = 300

        try:

            from PIL import Image as PILImage

            pil = PILImage.open(
                image_path
            )

            iw, ih = pil.size

            ratio = min(
                max_image_w / iw,
                max_image_h / ih
            )

            dw = iw * ratio
            dh = ih * ratio

            c.drawImage(
                ImageReader(pil),
                margin,
                y - dh,
                width=dw,
                height=dh,
                preserveAspectRatio=True,
                mask="auto"
            )

            y -= dh + 20

        except Exception:
            pass

        # Disease information.
        info = get_info(
            APP.result_name
        )

        c.setFont(
            "Helvetica-Bold",
            11
        )

        c.drawString(
            margin,
            y,
            "Cause / likely factors:"
        )

        y -= 15

        c.setFont(
            "Helvetica",
            9
        )

        y = self.pdf_wrap(
            c,
            info["cause"],
            margin,
            y,
            page_w - 2 * margin
        )

        y -= 10

        c.setFont(
            "Helvetica-Bold",
            11
        )

        c.drawString(
            margin,
            y,
            "Management / treatment:"
        )

        y -= 15

        c.setFont(
            "Helvetica",
            9
        )

        y = self.pdf_wrap(
            c,
            info["management"],
            margin,
            y,
            page_w - 2 * margin
        )

        y -= 10

        c.setFont(
            "Helvetica-Bold",
            11
        )

        c.drawString(
            margin,
            y,
            "Chemical options:"
        )

        y -= 15

        c.setFont(
            "Helvetica",
            9
        )

        y = self.pdf_wrap(
            c,
            info["chemicals"],
            margin,
            y,
            page_w - 2 * margin
        )

        y -= 20

        c.setFont(
            "Helvetica-Oblique",
            8
        )

        c.drawString(
            margin,
            y,
            "Safety: use only products registered for the crop and disease "
            "in your location. Follow the current product label."
        )

        c.save()


    @staticmethod
    def pdf_wrap(
        c,
        text,
        x,
        y,
        width
    ):

        words = text.split()
        line = ""

        # Approximate Helvetica character width.
        max_chars = max(
            40,
            int(width / 4.7)
        )

        for word in words:

            test = (
                line + " " + word
            ).strip()

            if len(test) > max_chars:

                c.drawString(
                    x,
                    y,
                    line
                )

                y -= 12

                line = word

            else:

                line = test

        if line:

            c.drawString(
                x,
                y,
                line
            )

            y -= 12

        return y


    # ============================================================
    # WEATHER DISPLAY
    # ============================================================

    def update_weather(self):

        temp, humidity = weather_text()

        self.weather_label.configure(
            text=(
                f"{tr('temperature')}: {temp}    "
                f"{tr('humidity')}: {humidity}    DHT11 / D8"
            )
        )

        self.root.after(
            1000,
            self.update_weather
        )


    # ============================================================
    # NAVIGATION
    # ============================================================

    def back_to_crops(self):

        self.stop_camera()
        self.show_crop_selection()


    def quit_app(self):

        self.stop_camera()

        try:
            DHT.stop()
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception:
            pass


    def run(self):

        self.root.mainloop()


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 72)
    print("CropAI // ONE AI // 5-CROP UNO Q")
    print("=" * 72)

    print(
        "Models directory:",
        MODEL_DIR
    )

    print(
        "Output directory:",
        OUTPUT_DIR
    )

    # Start DHT11 Bridge reader.
    DHT.start()

    app = CropAIApp()

    app.run()


if __name__ == "__main__":
    main()
