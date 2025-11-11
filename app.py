# ==============================================================================
#                      DOFUS TREASURE HUNT BOT (REMASTERED)
# ==============================================================================
#
#  Ce bot automatise les chasses au trésor sur Dofus en utilisant la
#  reconnaissance d'écran (OCR) et l'automatisation des clics et des saisies.
#
#  MOTEUR OCR UTILISÉ : EasyOCR
#
#  Dépendances (à installer avec 'pip install ...'):
#  - opencv-python      (uniquement pour lire les images, pas de traitement complexe)
#  - pyautogui          (pour contrôler la souris et le clavier)
#  - pyperclip          (pour la gestion du presse-papiers)
#  - Pillow             (dépendance de pyautogui)
#  - numpy              (dépendance d'easyocr)
#  - fuzzywuzzy         (pour la comparaison "floue" des indices)
#  - python-levenshtein (accélère fuzzywuzzy, recommandé)
#  - easyocr            (le moteur OCR)
#  - pynput             (uniquement pour le mode --calibrate)
#
# ==============================================================================

import cv2
import pyautogui
import time
import re
import json
import os
import pyperclip
import numpy as np
from fuzzywuzzy import fuzz
from PIL import Image
import easyocr
import winsound
import sys
import threading
import queue
try:
    from pynput.mouse import Listener as MouseListener, Button
    from pynput.keyboard import Listener as KeyListener, Key
except ImportError:
    MouseListener = None
    KeyListener = None
    Button = None
    Key = None

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:
    tk = None
    ttk = None
    messagebox = None

# Configure un délai global pour laisser le temps à l'interface de réagir
pyautogui.PAUSE = 0.1

# Ajout: événements pour le mode GUI
STOP_EVENT = threading.Event()
RESUME_EVENT = threading.Event()
GUI_MODE = False

def wait_or_stop(duration: float, stop_event: threading.Event | None = None, interval: float = 0.05) -> bool:
    """
    Attend 'duration' secondes par petits intervalles en vérifiant stop_event.
    Retourne True si un arrêt a été demandé, False sinon.
    """
    end = time.time() + duration
    while time.time() < end:
        if stop_event is not None and stop_event.is_set():
            return True
        time.sleep(interval)
    return False

def show_startup_message(mode: str = "cli"):
    """Affiche un message court au démarrage du bot."""
    banner = "=== Dofus Treasure Hunt Bot — prêt ==="
    tips = "Astuce: --calibrate pour configurer | --gui pour l'interface | Ctrl+C pour arrêter | Parametres recommandé :\n Options/Accessibilité/ taille des textes : Grand\n Echelle de l'interface 110 %\nOnglet chasse/ Taille de police : Petit\nSur Dofus DB ne pas oublier copier automatiquement le trajet \n Dans le chat du jeu mettez vous en mode message privé"
    when = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{banner}\n{tips}\nDémarré à: {when} ({mode})\n")
    try:
        winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
    except Exception:
        pass

# ==============================================================================
#                       LISTE DES INDICES DE CHASSE
# ==============================================================================
# Cette liste est utilisée pour corriger les résultats de l'OCR via une
# comparaison floue (fuzzy matching).
# ------------------------------------------------------------------------------
HINT_LIST = [
    "Affiche de carte au trésor", "Aiguille à coudre", "Ancre dorée", "Anneau d'or", "Arbre à épines",
    "Arbre à moitié coupé", "Arbre à trous", "Arbre arc-en-ciel", "Arbre ensanglanté", "Arbre glacé",
    "Arche naturelle", "Balançoire macabre", "Ballons en forme de coeur", "Bannière bontarienne déchirée",
    "Bannière brâkmarienne déchirée", "Barque coulée", "Blé noir et blanc", "Bombe coeur", "Bonbon bleu",
    "Bonhomme de neige fondu", "Bougie dans un trou", "Boule dorée de marin", "Bouton de couture",
    "Buisson poupée sadida", "Cactus à fleur bleue", "Cactus sans épines", "Cadran solaire", "Cairn",
    "Canard en plastique", "Canne à kebab", "Carapace de tortue", "Casque à cornes", "Ceinture cloutée",
    "Cerf-volant shinkansen", "Champignon rayé", "Chapeau dé", "Chaussette à pois", "Clef dorée",
    "Cocotte origami", "Coeur dans un nénuphar", "Coquillage à pois", "Corail avec des dents", "Corail flûtiste",
    "Corne de likrone", "Crâ cramé", "Crâne dans un trou", "Crâne de Crâ", "Crâne de cristal",
    "Crâne de likrone", "Crâne de likrone dans la glace", "Crâne de renne", "Crâne de Roublard",
    "Croix en pierre brisée", "Dé en glace", "Dessin au pipi dans la neige", "Dessin de croix dans un cercle",
    "Dessin dragodinde", "Dessin koalak", "Dofus en bois", "Dolmen", "Échelle cassée", "Éolienne à quatre pales",
    "Épée prise dans la glace", "Épouvantail à pipe", "Étoile en papier plié", "Étoile jaune peinte",
    "Étoile rouge en papier", "Étoile verte en papier", "Fer à cheval", "Filon de cristaux multicolores",
    "Flèche dans une pomme", "Fleur de nénuphar bleue", "Fleurs smiley", "Framboisier", "Girouette dragodinde",
    "Grand coquillage cassé", "Gravure d'aile", "Gravure d'arakne", "Gravure d'Epée", "Gravure d'étoile",
    "Gravure d'oeil", "Gravure de bouftou", "Gravure de boule de poche", "Gravure de chacha", "Gravure de clef",
    "Gravure de coeur", "Gravure de crâne", "Gravure de croix", "Gravure de Dofus", "Gravure de dragodinde",
    "Gravure de fantôme", "Gravure de Firefoux", "Gravure de flèche", "Gravure de fleur", "Gravure de Gelax",
    "Gravure de Kama", "Gravure de logo Ankama", "Gravure de lune", "Gravure de papatte", "Gravure de point",
    "Gravure de rose des vents", "Gravure de soleil", "Gravure de spirale", "Gravure de symbole de quête",
    "Gravure de symbole égal", "Gravure de tofu", "Gravure de wabbit", "Gravure de wukin", "Grelot",
    "Hache brisée", "Kaliptus à fleurs jaunes", "Kaliptus coupé", "Kaliptus grignoté", "Kama peint",
    "Lampadaire fungus", "Lampion bleu", "Langue dans un trou", "Lanterne au crâne luminescent",
    "Lapino origami", "Logo Ankama peint", "Marionnette", "Menottes", "Minouki", "Moufles jaunes",
    "Moufles rouges", "Niche dans une caisse", "Obélisque enfoui", "Oeil de shushu peint", "Oeuf dans un trou",
    "Ornement flocon", "Os dans la lave", "Paire de lunettes", "Palmier à feuilles carrées",
    "Palmier à feuilles déchirées", "Palmier à pois", "Palmier peint à rayures", "Palmier peint d'un chacha",
    "Palmier surchargé de noix de coco", "Palmifleur bleu fleuri", "Palmifleur jaune fleuri",
    "Palmifleur vert fleuri", "Panneau nonosse", "Peinture de Dofus", "Peluche de likrone",
    "Phorreur sournois / baveux / chafouin / fourbe", "Pioche plantée", "Plaque gravée d'un coeur",
    "Plaque gravée d'un crâne", "Plaque gravée d'un fantôme", "Plaque gravée d'un Kama",
    "Plaque gravée d'un logo Ankama", "Plaque gravée d'un oeil", "Plaque gravée d'un soleil",
    "Plaque gravée d'un symbole de quête", "Plaque gravée d'un symbole égal", "Plaque gravée d'un wukin",
    "Plaque gravée d'une Epée", "Plaque gravée d'une étoile", "Plaque gravée d'une flèche",
    "Plaque gravée d'une fleur", "Plaque gravée d'une lune", "Plaque gravée d'une papatte",
    "Poisson grillé embroché", "Poupée koalak", "Queue d'Osamodas", "Rocher à sédimentation verticale",
    "Rocher crâne", "Rocher dé", "Rocher Dofus", "Rocher percé", "Rocher taillé en arètes de poisson",
    "Rose des vents dorée", "Rose noire", "Rouage pris dans la glace", "Ruban bleu noué", "Rune nimbos",
    "Sapin couché", "Serrure dorée", "Sève qui s'écoule", "Slip à petit coeur", "Soupe de bananagrumes",
    "Squelette d'Ouginak pendu", "Statue koalak", "Statue sidoa", "Statue wabbit", "Stèle chacha",
    "Sucre d'orge", "Symbole de quête peint", "Talisman en papier", "Tambour à rayures", "Tambour papatte",
    "Théière à rayures", "Tissu à carreaux noué", "Tombe gravée d'un bouclier", "Tombe inondée",
    "Tombe inondée de sang", "Torii cassé", "Trace de main en sang", "Tricycle", "Trou mielleux",
    "Tube rempli de tofus"
]

# ==============================================================================
#                      GESTIONNAIRE DU MOTEUR OCR (EasyOCR)
# ==============================================================================
#
# Le "singleton" garantit que le modèle EasyOCR n'est chargé en mémoire qu'une
# seule fois au démarrage, ce qui rend les appels OCR suivants quasi-instantanés.
#
# ------------------------------------------------------------------------------

_easyocr_instance = None

def get_ocr_instance():
    """
    Initialise et retourne l'instance unique du lecteur EasyOCR.
    Utilise les modèles embarqués (easyocr_models) si disponibles.
    Désactive le GPU lorsqu'on est packagé (pour compatibilité).
    """
    global _easyocr_instance
    if _easyocr_instance is not None:
        return _easyocr_instance

    # Emplacement des données embarquées avec PyInstaller (onefile)
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "easyocr_models")

    # Si l'appli est "gelée" (packagée), force le CPU
    use_gpu = False if getattr(sys, "frozen", False) else True

    print("Initialisation d'EasyOCR...")
    try:
        # Tente d'utiliser les modèles embarqués sans téléchargement
        _easyocr_instance = easyocr.Reader(
            ['fr'],
            gpu=use_gpu,
            model_storage_directory=model_dir,
            download_enabled=False
        )
        print(f"EasyOCR prêt (gpu={'ON' if use_gpu else 'OFF'}; modèles={model_dir}).")
    except Exception as e:
        print(f"Échec du chargement des modèles embarqués: {e}")
        print("Nouvelle tentative avec téléchargement activé (réseau requis la 1ère fois)...")
        _easyocr_instance = easyocr.Reader(
            ['fr'],
            gpu=False,
            model_storage_directory=model_dir,
            download_enabled=True
        )
        print(f"EasyOCR prêt (gpu=OFF; modèles={model_dir}).")
    return _easyocr_instance


def ocr_text_from_image(image_or_path):
    """
    Extrait le texte d'une image en utilisant l'instance partagée d'EasyOCR.

    Args:
        image_or_path (str | np.ndarray): Chemin ou image (format numpy array).

    Returns:
        str: Le texte extrait, sinon une chaîne vide en cas d'erreur.
    """
    reader = get_ocr_instance()
    try:
        # detail=0 simplifie la sortie en une liste de chaînes de caractères.
        # paragraph=True aide à regrouper le texte de manière logique.
        result = reader.readtext(image_or_path, detail=0, paragraph=True)
        return " ".join(result).strip()
    except Exception as e:
        print(f"ERREUR OCR: {e}")
        return ""

# ==============================================================================
#                 FONCTIONS D'EXTRACTION (INDICES & COORDONNÉES)
# ==============================================================================

def extract_coordinates_from_image(image_path: str):
    """
    Extrait les coordonnées [X, Y] d'une image en utilisant l'OCR, avec un
    traitement d'image amélioré pour une meilleure reconnaissance des caractères,
    notamment le signe moins (-).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Échec du chargement de l'image : '{image_path}'")

    # --- AJOUTS ET MODIFICATIONS ---

    # 1. Agrandir l'image (facteur 4x pour encore plus de détails).
    #    INTER_LANCZOS4 est souvent considéré comme supérieur à CUBIC pour la netteté.
    img_processed = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)

    # 2. Appliquer le seuillage binaire pour obtenir une image N&B pure.
    _, img_processed = cv2.threshold(img_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. INVERSER L'IMAGE (CRUCIAL).
    #    Le texte de Dofus est blanc. L'OCR performe mieux avec du texte noir sur fond blanc.
    img_processed = cv2.bitwise_not(img_processed)

    # (Optionnel, pour le débogage) Sauvegarder l'image traitée pour voir ce que l'OCR voit.
    # cv2.imwrite('processed_coords.png', img_processed)
    
    # --- FIN DES AJOUTS ---

    raw_text = ocr_text_from_image(img_processed)
    
    print(f"Texte brut extrait par l'OCR (après traitement avancé) : '{raw_text}'")

    # ... le reste de votre code est parfait et reste inchangé ...
    if not raw_text:
        raise ValueError(f"Échec de l'OCR: Aucun texte n'a été extrait de '{image_path}'")

    corrections = {'O': '0', 'o': '0', 'S': '5', 's': '5', 'l': '1', 'Z': '2', 'B': '8', 'q': '9', 'g': '9'}
    corrected_text = raw_text
    for char, replacement in corrections.items():
        corrected_text = corrected_text.replace(char, replacement)

    corrected_text = re.sub(r'[—–−_]', '-', corrected_text)
    separators_replaced = re.sub(r'[^\d-]', ' ', corrected_text)
    separators_replaced = re.sub(r'-\s+', '-', separators_replaced)
    numbers = re.findall(r'-?\d+', separators_replaced)

    if len(numbers) >= 2:
        pos_x, pos_y = numbers[0], numbers[1]
        print(f"Coordonnées trouvées : X={pos_x}, Y={pos_y}")
        return pos_x, pos_y

    raise ValueError(f"Échec de l'extraction des coordonnées depuis le texte OCR : '{raw_text}' (traité en : '{separators_replaced}')")



def normalize_text(text: str) -> str:
    """Nettoie une chaîne de caractères pour optimiser la comparaison."""
    return text.lower().replace('œ', 'oe').replace("'", " ").replace("-", " ")


def find_best_hint_match(ocr_text: str, hint_list: list, min_score: int = 45) -> str:
    """
    Compare le texte de l'OCR avec la liste d'indices (HINT_LIST) et retourne
    la correspondance la plus probable.

    Args:
        ocr_text (str): Le texte de l'indice extrait par l'OCR.
        hint_list (list): La liste de tous les indices possibles.
        min_score (int): Le score de confiance minimum (0-100) pour valider une correspondance.

    Returns:
        str: L'indice le plus probable de la liste, ou le texte original si aucun match n'est trouvé.
    """
    if not ocr_text:
        return ""

    normalized_ocr = normalize_text(ocr_text)

    # Cas spécial pour "Phorreur" qui a plusieurs variantes.
    if 'phorreur' in normalized_ocr:
        return "Phorreur sournois / baveux / chafouin / fourbe"

    best_match = ""
    highest_score = 0

    # Calcule le score de similarité pour chaque indice de la liste.
    for hint in hint_list:
        score = fuzz.ratio(normalized_ocr, normalize_text(hint))
        if score > highest_score:
            highest_score = score
            best_match = hint

    print(f"Meilleur match pour '{ocr_text}': '{best_match}' (Score: {highest_score})")

    return best_match if highest_score >= min_score else ocr_text


def parse_hint_from_image(image_path: str):
    """
    Analyse l'image d'un indice pour en extraire l'élément et la direction,
    avec une logique améliorée et plus tolérante aux erreurs de l'OCR.

    Args:
        image_path (str): Le chemin de l'image de l'indice.

    Returns:
        tuple[str, str]: Un tuple (élément_corrigé, direction).
    """
    # Le traitement de l'image reste inchangé, il est déjà efficace.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erreur: Impossible de lire l'image de l'indice '{image_path}'")
        return "", ""

    try:
        img_processed = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        _, img_processed = cv2.threshold(img_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception as e:
        print(f"Avertissement: Le traitement de l'image de l'indice a échoué. Erreur : {e}")
        img_processed = img # Utilise l'image originale en cas d'échec

    raw_text = ocr_text_from_image(img_processed)
    print(f"Texte brut de l'indice : '{raw_text}'")

    # --- Extraction de la direction (inchangée) ---
    direction = ""
    text_lower = raw_text.lower()
    if "nord" in text_lower: direction = "nord"
    elif "sud" in text_lower: direction = "sud"
    elif "ouest" in text_lower or "louest" in text_lower: direction = "ouest"
    elif "est" in text_lower or "lest" in text_lower: direction = "est"

    # ==========================================================================
    #                 NOUVELLE LOGIQUE D'EXTRACTION DE L'ÉLÉMENT
    # ==========================================================================
    element = ""
    
    # Méthode 1 : Recherche améliorée du texte entre délimiteurs.
    # Cette regex tolère les erreurs de l'OCR comme [...) ou (...) au lieu de [...]
    # et gère les espaces après le crochet ouvrant ou avant le fermant.
    match = re.search(r'[\[\(]\s*([^\]\)]+?)\s*[\]\)]', raw_text)
    
    if match:
        element = match.group(1)
    else:
        # Méthode 2 (Fallback) : Si aucun délimiteur n'est trouvé, on se base sur
        # des mots-clés pour isoler l'indice. C'est plus fiable.
        # On sépare la phrase à partir de "jusqu'à" ou "jusqu'".
        parts = re.split(r"jusqu'à|jusqu'", raw_text, maxsplit=1, flags=re.IGNORECASE)
        
        if len(parts) > 1:
            element = parts[1] # On prend la partie qui suit le mot-clé.
        else:
            # Méthode 3 (Dernier recours) : Si rien n'a fonctionné, on tente
            # de supprimer la partie directionnelle avec une regex corrigée qui
            # gère "dirigez vous" et "dirigez-vous".
            element = re.sub(r"dirigez\s*-?\s*vous.*?(nord|sud|est|ouest)", "", raw_text, flags=re.I)

    # --- FIN DE LA NOUVELLE LOGIQUE ---

    # Nettoyage et correction de l'élément trouvé
    element = element.replace("\n", " ").strip()
    corrected_element = find_best_hint_match(element, HINT_LIST)

    # Affiche la correction uniquement si elle est significative
    if corrected_element != element and element.strip() != "":
        print(f"Correction d'indice : '{element}' -> '{corrected_element}'")

    return corrected_element, direction


# ==============================================================================
#                 CONFIGURATION ET CALIBRATION (INCHANGÉ)
# ==============================================================================
#
# Le mode --calibrate permet de configurer facilement les positions des clics
# et des captures d'écran pour s'adapter à votre résolution d'écran.
# Le fichier 'calibration.json' est créé pour sauvegarder ces réglages.
#
# ------------------------------------------------------------------------------

try:
    from pynput.mouse import Listener, Button
except ImportError:
    Listener, Button = None, None

CFG = {}
CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

class CalibrationManager:
    """Gestionnaire centralisé pour le chargement, la sauvegarde et la validation des calibrations."""
    
    def __init__(self, config_path=None):
        self.config_path = config_path or CFG_PATH
        self.backup_path = self.config_path + ".bak"
        self.config = {}
        
    def get_default_config(self):
        """Retourne la configuration par défaut."""
        return {
            "open_dofus": {"x": 1184, "y": 1051}, 
            "open_chrome": {"x": 1125, "y": 1053},
            "open_chasse_tab": {"x": 92, "y": 16}, 
            "dofus_chat_input": {"x": 186, "y": 1009},
            "starting_pos_region": {"x": 111, "y": 196, "w": 51, "h": 16},
            "current_pos_region": {"x": 16, "y": 76, "w": 78, "h": 28},
            "input_pos_x": {"x": 701, "y": 322}, 
            "input_pos_y": {"x": 1042, "y": 325},
            "dofus_ui_deadpoint": {"x": 539, "y": 16}, 
            "chasse_deadpoint": {"x": 220, "y": 1006},
            "chasse_direction_buttons": {
                "nord": {"x": 954, "y": 483}, 
                "est": {"x": 1018, "y": 503}, 
                "ouest": {"x": 898, "y": 517}, 
                "sud": {"x": 949, "y": 561}
            },
            "chasse_element_input": {"x": 1055, "y": 675}, 
            "chasse_element_option": {"x": 1055, "y": 716},
            "hint_hover_first": {"x": 79, "y": 230}, 
            "hint_region_first": {"x": 27, "y": 153, "w": 360, "h": 60},
            "hint_click_first": {"x": 301, "y": 230}, 
            "row_step": 30,
            "travel_confirm_button": {"x": 880, "y": 580},
            "validate_click_offset_y": -8
        }
    
    def validate_config(self, cfg):
        """Valide la structure de la configuration."""
        required_point_keys = [
            "open_dofus", "open_chrome", "open_chasse_tab", "dofus_chat_input",
            "input_pos_x", "input_pos_y", "dofus_ui_deadpoint", "chasse_deadpoint",
            "chasse_element_input", "chasse_element_option", "hint_hover_first",
            "hint_click_first", "travel_confirm_button"
        ]
        required_region_keys = ["starting_pos_region", "current_pos_region", "hint_region_first"]
        
        for key in required_point_keys:
            if key not in cfg or not isinstance(cfg[key], dict):
                return False
            if "x" not in cfg[key] or "y" not in cfg[key]:
                return False
        
        for key in required_region_keys:
            if key not in cfg or not isinstance(cfg[key], dict):
                return False
            if not all(k in cfg[key] for k in ["x", "y", "w", "h"]):
                return False
        
        if "chasse_direction_buttons" not in cfg:
            return False
        for dir_key in ["nord", "est", "ouest", "sud"]:
            if dir_key not in cfg["chasse_direction_buttons"]:
                return False
            btn = cfg["chasse_direction_buttons"][dir_key]
            if "x" not in btn or "y" not in btn:
                return False
        
        if "row_step" not in cfg or not isinstance(cfg["row_step"], (int, float)):
            return False
        
        return True
    
    def load(self):
        """Charge la configuration depuis le fichier JSON."""
        if not os.path.exists(self.config_path):
            print(f"Fichier de calibration non trouvé, création de la configuration par défaut.")
            self.config = self.get_default_config()
            self.save()
            return self.config
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded_cfg = json.load(f)
            
            if not self.validate_config(loaded_cfg):
                print("Configuration invalide, tentative de restauration depuis la sauvegarde...")
                if os.path.exists(self.backup_path):
                    with open(self.backup_path, "r", encoding="utf-8") as f:
                        loaded_cfg = json.load(f)
                    if self.validate_config(loaded_cfg):
                        print("Configuration restaurée depuis la sauvegarde.")
                        self.config = loaded_cfg
                        self.save()
                        return self.config
                
                print("Utilisation de la configuration par défaut.")
                self.config = self.get_default_config()
                self.save()
                return self.config
            
            self.config = loaded_cfg
            print(f"Configuration chargée: {len(self.config)} clés principales.")
            return self.config
            
        except json.JSONDecodeError as e:
            print(f"Erreur JSON dans la calibration: {e}")
            if os.path.exists(self.backup_path):
                print("Tentative de restauration depuis la sauvegarde...")
                try:
                    with open(self.backup_path, "r", encoding="utf-8") as f:
                        self.config = json.load(f)
                    print("Configuration restaurée depuis la sauvegarde.")
                    return self.config
                except Exception:
                    pass
            
            print("Utilisation de la configuration par défaut.")
            self.config = self.get_default_config()
            self.save()
            return self.config
        except Exception as e:
            print(f"Erreur lors du chargement de la calibration: {e}")
            self.config = self.get_default_config()
            self.save()
            return self.config
    
    def save(self):
        """Sauvegarde la configuration avec backup automatique."""
        if not self.config:
            print("Aucune configuration à sauvegarder.")
            return False
        
        if not self.validate_config(self.config):
            print("Configuration invalide, sauvegarde annulée.")
            return False
        
        try:
            if os.path.exists(self.config_path):
                try:
                    import shutil
                    shutil.copy2(self.config_path, self.backup_path)
                except Exception as e:
                    print(f"Avertissement: impossible de créer la sauvegarde: {e}")
            
            temp_path = self.config_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            if os.path.exists(self.config_path):
                os.remove(self.config_path)
            os.rename(temp_path, self.config_path)
            
            print(f"Configuration sauvegardée: {self.config_path}")
            return True
            
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            return False
    
    def export_to(self, path):
        """Exporte la configuration vers un fichier."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"Configuration exportée vers: {path}")
            return True
        except Exception as e:
            print(f"Erreur lors de l'export: {e}")
            return False
    
    def import_from(self, path):
        """Importe une configuration depuis un fichier."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                imported_cfg = json.load(f)
            
            if not self.validate_config(imported_cfg):
                print("Configuration importée invalide.")
                return False
            
            self.config = imported_cfg
            self.save()
            print(f"Configuration importée depuis: {path}")
            return True
        except Exception as e:
            print(f"Erreur lors de l'import: {e}")
            return False

_calib_manager = CalibrationManager()

def load_config():
    """Charge le fichier de configuration JSON."""
    global CFG
    CFG = _calib_manager.load()
    return CFG

def save_config():
    """Sauvegarde la configuration actuelle dans le fichier JSON."""
    global CFG
    _calib_manager.config = CFG
    return _calib_manager.save()

def _wait_for_click(prompt):
    """Fonction utilitaire pour le mode calibration (point).
    Nouveau: pressez 'c' pour capturer la position du curseur (ou clic gauche). Échap pour annuler."""
    if MouseListener is None or KeyListener is None:
        print("Le mode --calibrate requiert 'pynput'. Lancez : pip install pynput")
        raise RuntimeError("Calibration indisponible: 'pynput' manquant")
    print(f"[CALIBRATION] {prompt}\n  - Pressez 'c' pour capturer la position du curseur\n  - ou cliquez gauche\n  - Échap pour annuler")

    pos = {"x": None, "y": None}
    done = threading.Event()

    def on_click(x, y, button, pressed):
        if pressed and button == Button.left and not done.is_set():
            pos["x"], pos["y"] = int(x), int(y)
            done.set()
            return False
        
    def on_key(key):
        try:
            if key == Key.esc and not done.is_set():
                done.set()
                return False
            ch = getattr(key, "char", None)
            if ch and ch.lower() == "c" and not done.is_set():
                x, y = pyautogui.position()
                pos["x"], pos["y"] = int(x), int(y)
                done.set()
                return False
        except Exception:
            pass

    ml = MouseListener(on_click=on_click)
    kl = KeyListener(on_press=on_key)
    ml.start(); kl.start()
    done.wait()
    try: ml.stop()
    except Exception: pass
    try: kl.stop()
    except Exception: pass

    if pos["x"] is None:
        raise RuntimeError("Calibration annulée")
    print(f"  -> Enregistré : ({pos['x']}, {pos['y']})")
    time.sleep(0.2)
    return pos
    
def _wait_for_region_drag(prompt_title):
    """Sélection d'une région par glisser-déposer (mouse down -> drag -> release) avec overlay visuel.
    Échap pour annuler."""
    if MouseListener is None or KeyListener is None:
        print("Le mode --calibrate requiert 'pynput'. Lancez : pip install pynput")
        raise RuntimeError("Calibration indisponible: 'pynput' manquant")
    print(f"[CALIBRATION] {prompt_title}\n  - Cliquez-gauche, faites glisser pour tracer la zone, relâchez pour valider\n  - Échap pour annuler")

    data = {"start": None, "end": None, "dragging": False}
    done = threading.Event()
    
    # Création d'un overlay transparent avec tkinter
    if tk is not None:
        overlay = tk.Tk()
        overlay.attributes('-alpha', 0.3)
        overlay.attributes('-topmost', True)
        overlay.overrideredirect(True)
        overlay.configure(bg='cyan')
        overlay.withdraw()
        
        canvas = tk.Canvas(overlay, bg='cyan', highlightthickness=2, highlightbackground='blue')
        canvas.pack(fill='both', expand=True)

    def update_overlay():
        if tk is None or not data["dragging"] or data["start"] is None or data["end"] is None:
            return
        x1, y1 = data["start"]
        x2, y2 = data["end"]
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        overlay.geometry(f'{w}x{h}+{x}+{y}')
        overlay.deiconify()

    def on_click(x, y, button, pressed):
        if button != Button.left:
            return
        if pressed:
            data["start"] = (int(x), int(y))
            data["dragging"] = True
            if tk is not None:
                overlay.after(0, update_overlay)
        else:
            if data["start"] is not None and not done.is_set():
                data["end"] = (int(x), int(y))
                data["dragging"] = False
                if tk is not None:
                    overlay.withdraw()
                done.set()
                return False

    def on_move(x, y):
        if data["dragging"] and data["start"] is not None:
            data["end"] = (int(x), int(y))
            if tk is not None:
                overlay.after(0, update_overlay)

    def on_key(key):
        if key == Key.esc and not done.is_set():
            data["dragging"] = False
            if tk is not None:
                overlay.withdraw()
            done.set()
            return False
    
    ml = MouseListener(on_click=on_click, on_move=on_move)
    kl = KeyListener(on_press=on_key)
    ml.start(); kl.start()
    done.wait()
    try: ml.stop()
    except Exception: pass
    try: kl.stop()
    except Exception: pass
    if tk is not None:
        overlay.destroy()

    if not data["start"] or not data["end"]:
        raise RuntimeError("Calibration annulée")

    (x1, y1), (x2, y2) = data["start"], data["end"]
    x, y = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)
    region = {"x": x, "y": y, "w": w, "h": h}
    print(f"  -> Enregistré : x={x}, y={y}, w={w}, h={h}")
    time.sleep(0.2)
    return region


def run_calibration():
    """Lance le processus guidé de calibration."""
    load_config()
    print("\n=== MODE CALIBRATION ===")
    print("Veuillez capturer les éléments demandés (pressez 'c' pour un point, glissez-déposez pour une zone).")

    CFG["open_dofus"] = _wait_for_click("Placez le curseur sur l'icône Dofus (barre des tâches)")
    CFG["open_chrome"] = _wait_for_click("Placez le curseur sur l'icône Chrome (barre des tâches)")
    CFG["open_chasse_tab"] = _wait_for_click("Placez le curseur sur l'onglet de la chasse (dans Chrome)")
    CFG["dofus_chat_input"] = _wait_for_click("Placez le curseur sur la barre de saisie du chat (Dofus)")

    print("\nDéfinissez la zone de la position de départ de la chasse (glisser-déposer):")
    CFG["starting_pos_region"] = _wait_for_region_drag("Tracez la zone de position de départ (HUD)")

    print("\nDéfinissez la zone de votre position actuelle en jeu (glisser-déposer):")
    CFG["current_pos_region"] = _wait_for_region_drag("Tracez la zone de la position actuelle (HUD)")

    CFG["input_pos_x"] = _wait_for_click("Placez le curseur sur le champ de saisie 'X' (site)")
    CFG["input_pos_y"] = _wait_for_click("Placez le curseur sur le champ de saisie 'Y' (site)")
    CFG["dofus_ui_deadpoint"] = _wait_for_click("Placez le curseur sur un point neutre de l'UI Dofus")
    CFG["chasse_deadpoint"] = _wait_for_click("Placez le curseur sur un point neutre du site de chasse")

    CFG["chasse_direction_buttons"] = {
        "nord":  _wait_for_click("Placez le curseur sur le bouton 'Nord' (site)"),
        "est":   _wait_for_click("Placez le curseur sur le bouton 'Est' (site)"),
        "ouest": _wait_for_click("Placez le curseur sur le bouton 'Ouest' (site)"),
        "sud":   _wait_for_click("Placez le curseur sur le bouton 'Sud' (site)")
    }

    CFG["chasse_element_input"]  = _wait_for_click("Placez le curseur sur le champ de recherche d'indice (site)")
    CFG["chasse_element_option"] = _wait_for_click("Placez le curseur sur la 1ère option d'indice (site)")
    CFG["hint_hover_first"] = _wait_for_click("Placez le curseur au centre de la première ligne d'indice (Dofus)")

    print("\nDéfinissez la zone de capture pour le premier indice (glisser-déposer):")
    CFG["hint_region_first"] = _wait_for_region_drag("Tracez la zone de l'infobulle du 1er indice (Dofus)")

    CFG["hint_click_first"] = _wait_for_click("Placez le curseur sur le jalon (coche) de la première ligne (Dofus)")

    print("\nCalcule de l'espacement vertical des indices :")
    p1 = _wait_for_click("Placez le curseur sur le jalon de la PREMIÈRE ligne d'indice")
    p2 = _wait_for_click("Placez le curseur sur le jalon de la DEUXIÈME ligne d'indice")
    CFG["row_step"] = abs(p2['y'] - p1['y'])

    CFG["travel_confirm_button"] = _wait_for_click("Tapez /travel -10,10 dans Dofus, puis placez le curseur sur le bouton 'Oui'")

    save_config()
    print("\nCalibration terminée avec succès !")


# ==============================================================================
#                      ACTIONS ET AUTOMATISATION
# ==============================================================================

def open_dofus():
    """Active la fenêtre de Dofus en cliquant sur son icône dans la barre des tâches."""
    pyautogui.click(CFG["open_dofus"]["x"], CFG["open_dofus"]["y"])
    time.sleep(0.5)

def open_chrome_treasure_tab():
    """Active la fenêtre Chrome et sélectionne l'onglet de la chasse."""
    pyautogui.click(CFG["open_chrome"]["x"], CFG["open_chrome"]["y"])
    time.sleep(0.5)
    pyautogui.click(CFG["open_chasse_tab"]["x"], CFG["open_chasse_tab"]["y"])
    time.sleep(0.25)

def get_hint_screenshot(index: int):
    """
    Passe la souris sur un indice dans Dofus pour faire apparaître l'infobulle
    et prend une capture d'écran de celle-ci.

    Args:
        index (int): L'index de l'indice (0 pour le premier, 1 pour le deuxième, etc.).
    """
    step = int(CFG["row_step"])
    base_hover_pos = CFG["hint_hover_first"]
    base_region = CFG["hint_region_first"]

    pyautogui.moveTo(base_hover_pos["x"], base_hover_pos["y"] + step * index)
    time.sleep(0.3) 

    screenshot_path = f'hint_{index}.png'
    pyautogui.screenshot(
        screenshot_path,
        region=(base_region["x"], base_region["y"] + step * index, base_region["w"], base_region["h"])
    )
    return screenshot_path

def look_for_hint_on_website(start_pos_x: str, start_pos_y: str, direction: str, element: str, stop_event: threading.Event | None = None, is_first_hint: bool = False):
    """
    Automatise la recherche de l'indice suivant sur le site web de chasse.
    Retourne True si une commande /travel valide a été trouvée et envoyée, sinon False.
    
    Args:
        start_pos_x: Coordonnée X de départ
        start_pos_y: Coordonnée Y de départ
        direction: Direction à chercher
        element: Élément/indice à chercher
        stop_event: Événement d'arrêt
        is_first_hint: Si True, tape tous les champs (position, direction, indice).
                       Si False, ne tape que la direction (les indices sont gardés en mémoire par DofusDB).
    """
    open_chrome_treasure_tab()
    if stop_event is not None and stop_event.is_set(): 
        return False

    # OPTIMISATION: Ne taper position X/Y et l'indice QUE pour le premier indice
    # ou en mode récupération (is_first_hint=True)
    if is_first_hint:
        pyautogui.click(CFG["input_pos_x"]["x"], CFG["input_pos_x"]["y"])
        pyautogui.hotkey('ctrl', 'a'); pyautogui.press('delete')
        pyautogui.write(str(start_pos_x))

        if stop_event is not None and stop_event.is_set(): 
            return False
        pyautogui.click(CFG["input_pos_y"]["x"], CFG["input_pos_y"]["y"])
        pyautogui.hotkey('ctrl', 'a'); pyautogui.press('delete')
        pyautogui.write(str(start_pos_y))

        pyautogui.click(CFG["chasse_deadpoint"]["x"], CFG["chasse_deadpoint"]["y"])
        if wait_or_stop(0.25, stop_event): 
            return False

    # La direction doit toujours être cliquée
    if direction in CFG["chasse_direction_buttons"]:
        btn = CFG["chasse_direction_buttons"][direction]
        print(f"Direction sélectionnée : {direction.upper()}")
        pyautogui.click(btn["x"], btn["y"])
        if wait_or_stop(1, stop_event): 
            return False

    # L'indice doit TOUJOURS être tapé (car il change à chaque étape)
    pyautogui.click(CFG["chasse_element_input"]["x"], CFG["chasse_element_input"]["y"])
    pyautogui.hotkey('ctrl', 'a'); pyautogui.press('delete')

    if is_first_hint:
        print(f"Recherche de l'indice : '{element}'")
    else:
        print(f"Recherche de l'indice (positions gardées en mémoire) : '{element}'")
    
    pyperclip.copy(str(element).strip())
    pyautogui.hotkey('ctrl', 'v')
    if wait_or_stop(1.5, stop_event): 
        return False

    # Sélection de la première option (si présente)
    pyautogui.click(CFG["chasse_element_option"]["x"], CFG["chasse_element_option"]["y"])
    if wait_or_stop(0.6, stop_event): 
        return False

    # Copie de la commande (le site doit avoir copié automatiquement /travel ...)
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(0.1)
    travel_command = pyperclip.paste().strip()
    print(f"Commande récupérée : '{travel_command}'")

    # Validation stricte
    if not travel_command.lower().startswith("/travel"):
        print("ERREUR: Aucune commande /travel générée (indice introuvable ?). Rien n'est envoyé.")
        return False

    nums = re.findall(r'-?\d+', travel_command)
    if len(nums) < 2:
        print(f"ERREUR: Commande /travel invalide: '{travel_command}'. Annulation de l'envoi.")
        return False

    # OK -> envoi dans le chat
    open_dofus()
    if stop_event is not None and stop_event.is_set(): 
        return False
    pyautogui.click(CFG["dofus_chat_input"]["x"], CFG["dofus_chat_input"]["y"])
    pyautogui.hotkey('ctrl', 'a'); pyautogui.press('delete')
    pyautogui.write(travel_command)
    pyautogui.press('enter')
    if wait_or_stop(0.5, stop_event): 
        return False
    pyautogui.click(CFG["travel_confirm_button"]["x"], CFG["travel_confirm_button"]["y"])
    return True

def get_destination_coords_from_clipboard() -> tuple[str, str]:
    """
    Lit le presse-papiers pour extraire les coordonnées de destination.
    """
    clipboard_content = pyperclip.paste().strip()
    print(f"Contenu du presse-papiers (destination) : {clipboard_content}")
    numbers = re.findall(r'-?\d+', clipboard_content)
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    print("AVERTISSEMENT: Impossible de trouver les coordonnées de destination dans le presse-papiers.")
    return None, None

def get_current_pos_from_screen(timeout: int = 15, black_threshold: int = 20, stop_event: threading.Event | None = None) -> tuple[str, str]:
    """
    Attend que la carte se charge et capture la position actuelle du joueur.
    """
    r = CFG["current_pos_region"]
    start_time = time.time()
    screenshot_path = 'current_pos.png'

    print("Attente du chargement de la carte...")
    while time.time() - start_time < timeout:
        if stop_event is not None and stop_event.is_set():
            print("Arrêt demandé pendant la lecture de la position actuelle.")
            return None, None
        img = pyautogui.screenshot(region=(r["x"], r["y"], r["w"], r["h"]))
        mean_brightness = np.mean(np.array(img))

        if mean_brightness > black_threshold:
            print(f"Carte chargée (luminosité: {mean_brightness:.2f}). Analyse de la position...")
            img.save(screenshot_path)
            try:
                return extract_coordinates_from_image(screenshot_path)
            except ValueError as e:
                print(e)
        time.sleep(0.2)

    print(f"TIMEOUT: La carte ne semble pas s'être chargée après {timeout}s. Tentative d'analyse quand même.")
    pyautogui.screenshot(screenshot_path, region=(r["x"], r["y"], r["w"], r["h"]))
    try:
        return extract_coordinates_from_image(screenshot_path)
    except ValueError as e:
        print(f"Échec critique de la lecture de la position actuelle : {e}")
        return None, None

# ==============================================================================
#                               BOUCLE PRINCIPALE
# ==============================================================================

def run_bot(stop_event: threading.Event | None = None,
            resume_event: threading.Event | None = None,
            gui_mode: bool = False,
            start_pos: tuple[str, str] | None = None,
            start_hint_index: int | None = None):
    """Fonction principale qui exécute la logique du bot."""
    open_dofus()
    hint_index = start_hint_index if start_hint_index is not None else 0
    clicked_validate_for_stage = False  # Nouveau: éviter de cliquer en boucle sur "Validée"
    is_first_hint = True  # OPTIMISATION: Pour savoir si on doit tout taper sur DofusDB
    after_phorreur = False  # OPTIMISATION: Pour savoir si on vient de résoudre un Phorreur

    print("--- DÉBUT DE LA CHASSE ---")
    if start_pos is None:
        start_pos_region = CFG["starting_pos_region"]
        pyautogui.screenshot('starting_pos.png', region=(start_pos_region["x"], start_pos_region["y"], start_pos_region["w"], start_pos_region["h"]))
        try:
            pos_x, pos_y = extract_coordinates_from_image('starting_pos.png')
        except ValueError as e:
            print(f"ERREUR CRITIQUE: Impossible de lire la position de départ. {e}")
            return
    else:
        pos_x, pos_y = start_pos
        print(f"Mode récupération: départ manuel à [{pos_x}, {pos_y}], indice #{(hint_index + 1)}")
        is_first_hint = True  # En mode récupération, on doit tout retaper

    while True:
        if stop_event is not None and stop_event.is_set():
            print("Arrêt demandé. Fin de la chasse.")
            break

        print(f"\n--- ÉTAPE {hint_index + 1} ---")
        print(f"Position de départ pour cette étape : [{pos_x}, {pos_y}]")

        pyautogui.click(CFG["dofus_ui_deadpoint"]["x"], CFG["dofus_ui_deadpoint"]["y"])
        hint_screenshot_path = get_hint_screenshot(hint_index)
        element, direction = parse_hint_from_image(hint_screenshot_path)

        if not element or not direction:
            if not clicked_validate_for_stage:
                print("\nFIN DE L'ETAPE DÉTECTÉE (indice ou direction manquant).")
                # Clique sur la case "Validée" (une ligne EN DESSOUS du dernier jalon)
                step_px = int(CFG["row_step"])
                base_click_pos = CFG["hint_click_first"]
                offset = int(CFG.get("validate_click_offset_y", 0))
                pyautogui.click(
                    base_click_pos["x"],
                    base_click_pos["y"] + (step_px) * (hint_index + 1) + offset
                )
                if wait_or_stop(0.8, stop_event): return
                # Prépare la nouvelle étape
                hint_index = 0
                clicked_validate_for_stage = True
                is_first_hint = True  # OPTIMISATION: Nouvelle étape = tout retaper
                new_x, new_y = get_current_pos_from_screen(stop_event=stop_event)
                if new_x is not None:
                    pos_x, pos_y = new_x, new_y
                    print(f"Nouvelle étape: position de départ mise à jour [{pos_x}, {pos_y}]")
                    continue
                else:
                    print("Impossible de lire la position actuelle pour la nouvelle étape.")
                    try:
                        winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
                    except Exception:
                        pass
                    break
            else:
                print("Aucun nouvel indice détecté après validation. Arrêt du bot.")
                try:
                    winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
                except Exception:
                    pass
                break

        # On a un indice valide: on réinitialise le flag anti-boucle si nécessaire
        if clicked_validate_for_stage:
            clicked_validate_for_stage = False

        print(f"Indice trouvé : Aller vers '{direction.upper()}' en cherchant '{element}'")

        if 'Phorreur' in element:
            winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
            print("\nATTENTION : Phorreur détecté. Le bot se met en pause.")
            if gui_mode and resume_event is not None:
                print("Trouvez le Phorreur manuellement, puis cliquez sur 'Continuer' dans la fenêtre.")
                print("[GUI]ENABLE_CONTINUE")
                resume_event.clear()
                while not resume_event.is_set():
                    if stop_event is not None and stop_event.is_set():
                        print("Arrêt demandé pendant la pause Phorreur.")
                        return
                    time.sleep(0.2)
            else:
                print("Trouvez le Phorreur manuellement, puis appuyez sur 'Entrée' pour continuer...")
                input()

            # Valider la ligne courante (jalon)
            step = int(CFG["row_step"])
            base_click_pos = CFG["hint_click_first"]
            pyautogui.click(base_click_pos["x"], base_click_pos["y"] + step * hint_index)
            if wait_or_stop(0.5, stop_event): return

            # NOUVEAU: actualiser la position après la résolution du Phorreur
            print("Lecture de la position actuelle après résolution du Phorreur...")
            new_x, new_y = get_current_pos_from_screen(stop_event=stop_event)
            if new_x is not None:
                pos_x, pos_y = new_x, new_y
                print(f"Position mise à jour après Phorreur: [{pos_x}, {pos_y}]")
            else:
                print("Impossible de mettre à jour la position (OCR). Position précédente conservée.")

            # OPTIMISATION: Après un Phorreur, il faut tout retaper sur DofusDB
            is_first_hint = True
            hint_index += 1
            continue

        look_success = look_for_hint_on_website(pos_x, pos_y, direction, element, stop_event=stop_event, is_first_hint=is_first_hint)
        # OPTIMISATION: Après le premier indice, on ne retape plus les indices
        if is_first_hint:
            is_first_hint = False
        if not look_success:
            print("Aucune progression possible pour cet indice (pas de commande /travel). Interruption.")
            winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
            break
        if stop_event is not None and stop_event.is_set():
            print("Arrêt demandé. Fin de la chasse.")
            break
        if wait_or_stop(1, stop_event): return
        dest_x, dest_y = get_destination_coords_from_clipboard()

        if dest_x is None:
            print("ERREUR: Impossible de continuer sans coordonnées de destination.")
            break

        while True:
            if stop_event is not None and stop_event.is_set():
                print("Arrêt demandé pendant le déplacement.")
                return
            current_x, current_y = get_current_pos_from_screen(stop_event=stop_event)
            if current_x is None:
                print("Abandon de la vérification de la position.")
                break

            print(f"Position actuelle: [{current_x}, {current_y}] | Destination: [{dest_x}, {dest_y}]")
            if current_x == dest_x and current_y == dest_y:
                print("Arrivé à destination !")
                break
            time.sleep(2)

        step = int(CFG["row_step"])
        base_click_pos = CFG["hint_click_first"]
        pyautogui.click(base_click_pos["x"], base_click_pos["y"] + step * hint_index)
        if wait_or_stop(1, stop_event): return

        pos_x, pos_y = dest_x, dest_y
        hint_index += 1

# ==============================================================================
#                             POINT D'ENTRÉE DU SCRIPT
# ==============================================================================

def main():
    """Point d'entrée principal avec gestion des arguments CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dofus Treasure Hunt Bot")
    parser.add_argument("--calibrate", action="store_true", help="Lance le mode calibration CLI interactif")
    parser.add_argument("--gui", action="store_true", help="Lance l'interface graphique (par défaut)")
    parser.add_argument("--export", type=str, metavar="FILE", help="Exporte la calibration vers un fichier JSON")
    parser.add_argument("--import", type=str, dest="import_file", metavar="FILE", help="Importe une calibration depuis un fichier JSON")
    parser.add_argument("--validate", action="store_true", help="Valide la configuration actuelle")
    parser.add_argument("--reset", action="store_true", help="Réinitialise la configuration aux valeurs par défaut")
    
    args = parser.parse_args()
    
    if args.export:
        load_config()
        if _calib_manager.export_to(args.export):
            print(f"✓ Configuration exportée avec succès vers: {args.export}")
        else:
            print("✗ Erreur lors de l'export")
            sys.exit(1)
        return
    
    if args.import_file:
        if _calib_manager.import_from(args.import_file):
            print(f"✓ Configuration importée avec succès depuis: {args.import_file}")
        else:
            print("✗ Erreur lors de l'import ou configuration invalide")
            sys.exit(1)
        return
    
    if args.validate:
        cfg = load_config()
        if _calib_manager.validate_config(cfg):
            print("✓ Configuration valide")
            print(f"  - Nombre de clés: {len(cfg)}")
            print(f"  - Fichier: {CFG_PATH}")
            if os.path.exists(_calib_manager.backup_path):
                print(f"  - Sauvegarde disponible: {_calib_manager.backup_path}")
        else:
            print("✗ Configuration invalide")
            sys.exit(1)
        return
    
    if args.reset:
        print("Réinitialisation de la configuration...")
        _calib_manager.config = _calib_manager.get_default_config()
        if _calib_manager.save():
            print("✓ Configuration réinitialisée aux valeurs par défaut")
        else:
            print("✗ Erreur lors de la réinitialisation")
            sys.exit(1)
        return
    
    if args.calibrate:
        try:
            run_calibration()
        except Exception as e:
            print(f"Erreur pendant la calibration: {e}")
            sys.exit(1)
        return
    
    load_config()
    launch_gui()

# ============== Interface graphique (Tkinter) ==============

class _QueueWriter:
    def __init__(self, q: queue.Queue):
        self.q = q
    def write(self, s: str):
        if not s:
            return
        self.q.put(s)
    def flush(self):
        pass

# === Nouveau: Gestionnaire de calibration (GUI) ===
def open_calibration_manager(parent):
    if tk is None or ttk is None:
        print("Tkinter indisponible.")
        return
    load_config()

    if MouseListener is None or KeyListener is None:
        messagebox.showerror("Calibration", "Le mode calibration requiert 'pynput'.\nExécutez: pip install pynput")
        return

    win = tk.Toplevel(parent)
    win.title("Gestionnaire de calibration")
    win.geometry("770x520")
    win.resizable(False, False)
    try:
        win.wm_attributes("-topmost", True)
    except Exception:
        pass

    info = ttk.Label(win, text="Pressez 'c' pour capturer un point. Pour une zone, cliquez-glissez puis relâchez. (Échap pour annuler)", foreground="#555")
    info.pack(fill="x", padx=10, pady=(10, 0))

    main = ttk.Frame(win, padding=10)
    main.pack(fill="both", expand=True)

    # Définition des items
    # type: point | region | row_step | dir
    items = [
        ("open_dofus", "Icône Dofus (barre des tâches)", "point"),
        ("open_chrome", "Icône Chrome (barre des tâches)", "point"),
        ("open_chasse_tab", "Onglet chasse (Chrome)", "point"),
        ("dofus_chat_input", "Champ chat Dofus", "point"),
        ("starting_pos_region", "Région position de départ (HUD)", "region"),
        ("current_pos_region", "Région position actuelle (HUD)", "region"),
        ("input_pos_x", "Champ X (site de chasse)", "point"),
        ("input_pos_y", "Champ Y (site de chasse)", "point"),
        ("dofus_ui_deadpoint", "Point neutre UI Dofus", "point"),
        ("chasse_deadpoint", "Point neutre site de chasse", "point"),
        ("dir_nord", "Bouton direction Nord (site)", "dir"),
        ("dir_est", "Bouton direction Est (site)", "dir"),
        ("dir_ouest", "Bouton direction Ouest (site)", "dir"),
        ("dir_sud", "Bouton direction Sud (site)", "dir"),
        ("chasse_element_input", "Champ recherche d'indice (site)", "point"),
        ("chasse_element_option", "1ère option d'indice (site)", "point"),
        ("hint_hover_first", "Souris sur 1ère ligne d'indice (Dofus)", "point"),
        ("hint_region_first", "Région infobulle 1er indice (Dofus)", "region"),
        ("hint_click_first", "Jalon 1ère ligne (Dofus)", "point"),
        ("row_step", "Espacement vertical entre lignes (2 clics)", "row_step"),
        ("travel_confirm_button", "Bouton 'Oui' de /travel (Dofus)", "point"),
        ("validate_click_offset_y", "Offset clic 'Validée' (optionnel)", "offset"),
    ]

    # Helpers d’affichage
    def fmt_point(p):
        return f"({p.get('x','?')}, {p.get('y','?')})"
    def fmt_region(r):
        return f"x={r.get('x','?')}, y={r.get('y','?')}, w={r.get('w','?')}, h={r.get('h','?')}"
    def get_display_value(key, typ):
        if typ == "point":
            return fmt_point(CFG.get(key, {}))
        if typ == "region":
            return fmt_region(CFG.get(key, {}))
        if typ == "dir":
            dir_key = key.split("_", 1)[1]  # nord/est/ouest/sud
            return fmt_point(CFG.get("chasse_direction_buttons", {}).get(dir_key, {}))
        if typ == "row_step":
            return str(CFG.get("row_step", "?"))
        if typ == "offset":
            return str(CFG.get("validate_click_offset_y", "?"))
        return "?"

    # UI: liste + détails + boutons
    left = ttk.Frame(main)
    left.pack(side="left", fill="both", expand=True)
    right = ttk.Frame(main)
    right.pack(side="right", fill="y")

    columns = ("label", "value")
    tree = ttk.Treeview(left, columns=columns, show="headings", height=18)
    tree.heading("label", text="Élément")
    tree.heading("value", text="Valeur actuelle")
    tree.column("label", width=360, anchor="w")
    tree.column("value", width=280, anchor="w")
    tree.pack(side="left", fill="both", expand=True)

    scroll = ttk.Scrollbar(left, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")

    # Charge la table
    def reload_table():
        for i in tree.get_children():
            tree.delete(i)
        for key, label, typ in items:
            tree.insert("", "end", iid=key, values=(label, get_display_value(key, typ)))
    reload_table()

    # Détails
    detail_lbl = ttk.Label(right, text="Détails", font=("Segoe UI", 10, "bold"))
    detail_lbl.pack(anchor="w", pady=(0, 6))

    detail_text = tk.Text(right, height=14, width=36, wrap="word", state="disabled")
    detail_text.pack(fill="y")

    def set_detail(msg):
        detail_text.configure(state="normal")
        detail_text.delete("1.0", "end")
        detail_text.insert("end", msg)
        detail_text.configure(state="disabled")

    def on_select(_e=None):
        sel = tree.selection()
        if not sel:
            set_detail("")
            return
        key = sel[0]
        typ = next((t for k, _l, t in items if k == key), None)
        val = get_display_value(key, typ)
        help_text = ""
        if typ == "point":
            help_text = "Pressez 'c' pour capturer la position du curseur (ou cliquez gauche). Échap pour annuler."
        elif typ == "region":
            help_text = "Cliquez-gauche, faites glisser pour tracer la zone, relâchez pour valider. Échap pour annuler."
        elif typ == "dir":
            help_text = "Placez le curseur sur le bouton et pressez 'c' (ou cliquez gauche)."
        elif typ == "row_step":
            help_text = "Pressez 'c' sur le jalon de la ligne 1 puis sur le jalon de la ligne 2."
        set_detail(f"Clé: {key}\nType: {typ}\nValeur: {val}\n\n{help_text}")

    tree.bind("<<TreeviewSelect>>", on_select)

    # Actions de calibration (lancées en thread pour ne pas bloquer l'UI)
    busy = {"v": False}
    def run_async(fn):
        if busy["v"]:
            return
        busy["v"] = True
        def _wrap():
            try:
                fn()
            finally:
                busy["v"] = False
        threading.Thread(target=_wrap, daemon=True).start()

    def hide_win():
        try:
            win.after(0, win.withdraw)
        except Exception:
            pass
        time.sleep(0.25)

    def show_win():
        try:
            win.after(0, win.deiconify)
            win.after(0, lambda: win.wm_attributes("-topmost", True))
        except Exception:
            pass

    def calib_point(key, prompt):
        def work():
            print(f"[CALIBRATION] {prompt}")
            hide_win()
            try:
                pos = _wait_for_click(prompt)
                CFG[key] = pos
                save_config()
                parent.after(0, lambda: (reload_table(), on_select()))
            finally:
                show_win()
        run_async(work)

    def calib_region(key, prompt):
        def work():
            print(f"[CALIBRATION] {prompt}")
            hide_win()
            try:
                reg = _wait_for_region_drag(prompt)
                CFG[key] = reg
                save_config()
                parent.after(0, lambda: (reload_table(), on_select()))
            finally:
                show_win()
        run_async(work)

    def calib_dir(dir_key, prompt):
        def work():
            print(f"[CALIBRATION] {prompt}")
            hide_win()
            try:
                pos = _wait_for_click(prompt)
                CFG.setdefault("chasse_direction_buttons", {})
                CFG["chasse_direction_buttons"][dir_key] = pos
                save_config()
                parent.after(0, lambda: (reload_table(), on_select()))
            finally:
                show_win()
        run_async(work)

    def calib_row_step():
        def work():
            print("[CALIBRATION] Espacement vertical (row_step)")
            hide_win()
            try:
                p1 = _wait_for_click("Pressez 'c' sur le jalon de la PREMIÈRE ligne d'indice")
                p2 = _wait_for_click("Pressez 'c' sur le jalon de la DEUXIÈME ligne d'indice")
                CFG["row_step"] = abs(p2["y"] - p1["y"])
                save_config()
                parent.after(0, lambda: (reload_table(), on_select()))
            finally:
                show_win()
        run_async(work)

    def calib_offset():
        def work():
            print("[CALIBRATION] Offset clic 'Validée'")
            hide_win()
            try:
                p1 = _wait_for_click("Pressez 'c' à une première position")
                p2 = _wait_for_click("Pressez 'c' à une seconde position (au-dessus ou en dessous)")
                CFG["validate_click_offset_y"] = (p2["y"] - p1["y"])
                save_config()
                parent.after(0, lambda: (reload_table(), on_select()))
            finally:
                show_win()
        run_async(work)

    def do_set_selected():
        sel = tree.selection()
        if not sel:
            return
        key = sel[0]
        typ = next((t for k, _l, t in items if k == key), None)
        label = next((l for k, l, _t in items if k == key), key)

        if typ == "point":
            calib_point(key, f"{label}")
        elif typ == "region":
            calib_region(key, f"{label}")
        elif typ == "dir":
            dir_key = key.split("_", 1)[1]
            calib_dir(dir_key, f"{label}")
        elif typ == "row_step":
            calib_row_step()
        elif typ == "offset":
            calib_offset()
        else:
            print(f"Type inconnu: {typ}")
            return
        parent.after(0, lambda: (reload_table(), on_select()))

        def work():
            print(f"[CALIBRATION] {label}")
            if typ == "point":
                calib_point(key, f"{label} — cliquez gauche pour enregistrer")
            elif typ == "region":
                calib_region(key,
                             f"{label} — coin HAUT-GAUCHE",
                             f"{label} — coin BAS-DROIT")
            elif typ == "dir":
                dir_key = key.split("_", 1)[1]
                calib_dir(dir_key, f"{label} — cliquez sur le bouton")
            elif typ == "row_step":
                calib_row_step()
            elif typ == "offset":
                calib_offset()
            else:
                print(f"Type inconnu: {typ}")
                return
            # UI update
            parent.after(0, lambda: (reload_table(), on_select()))

        run_async(work)

    btn_set = ttk.Button(right, text="Définir / Recalibrer", command=do_set_selected)
    btn_set.pack(fill="x", pady=(10, 4))

    def do_reload():
        load_config()
        reload_table()
        on_select()

    btn_reload = ttk.Button(right, text="Recharger", command=do_reload)
    btn_reload.pack(fill="x", pady=(0, 4))

    def do_export():
        try:
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(
                parent=win,
                title="Exporter la calibration",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile="calibration_export.json"
            )
            if path:
                if _calib_manager.export_to(path):
                    messagebox.showinfo("Export", f"Configuration exportée vers:\n{path}", parent=win)
                else:
                    messagebox.showerror("Export", "Erreur lors de l'export", parent=win)
        except Exception as e:
            messagebox.showerror("Export", f"Erreur: {e}", parent=win)

    btn_export = ttk.Button(right, text="Exporter...", command=do_export)
    btn_export.pack(fill="x", pady=(0, 4))

    def do_import():
        try:
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                parent=win,
                title="Importer une calibration",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if path:
                if _calib_manager.import_from(path):
                    messagebox.showinfo("Import", f"Configuration importée depuis:\n{path}", parent=win)
                    do_reload()
                else:
                    messagebox.showerror("Import", "Configuration invalide ou erreur lors de l'import", parent=win)
        except Exception as e:
            messagebox.showerror("Import", f"Erreur: {e}", parent=win)

    btn_import = ttk.Button(right, text="Importer...", command=do_import)
    btn_import.pack(fill="x", pady=(0, 4))

    def do_reset():
        if messagebox.askyesno("Réinitialisation", 
                               "Réinitialiser la configuration aux valeurs par défaut?\n\nUne sauvegarde sera créée automatiquement.",
                               parent=win):
            global CFG
            CFG = _calib_manager.get_default_config()
            _calib_manager.config = CFG
            _calib_manager.save()
            do_reload()
            messagebox.showinfo("Réinitialisation", "Configuration réinitialisée aux valeurs par défaut.", parent=win)

    btn_reset = ttk.Button(right, text="Réinitialiser", command=do_reset)
    btn_reset.pack(fill="x", pady=(0, 4))

    def do_close():
        win.destroy()

    btn_close = ttk.Button(right, text="Fermer", command=do_close)
    btn_close.pack(fill="x", pady=(0, 4))

    # Sélection initiale
    try:
        tree.selection_set(items[0][0])
        on_select()
    except Exception:
        pass

def launch_gui():
    global GUI_MODE
    if tk is None or ttk is None:
        print("Tkinter indisponible. Lancez sans --gui.")
        return
    GUI_MODE = True

    root = tk.Tk()
    root.title("I love bot")
    root.geometry("520x560")
    root.resizable(False, False)

    topmost_var = tk.BooleanVar(value=True)
    root.wm_attributes("-topmost", True)
    def _toggle_topmost():
        root.wm_attributes("-topmost", topmost_var.get())

    # Contrôles
    btn_frame = ttk.Frame(root, padding=10)
    btn_frame.pack(fill="x")

    start_btn = ttk.Button(btn_frame, text="Lancer le bot")
    stop_btn = ttk.Button(btn_frame, text="Arrêter", state="disabled")
    calib_btn = ttk.Button(btn_frame, text="Calibration")
    cont_btn = ttk.Button(btn_frame, text="Continuer (Phorreur)", state="disabled")

    start_btn.pack(side="left", padx=(0,6))
   
    stop_btn.pack(side="left", padx=(0,6))
    calib_btn.pack(side="left", padx=(0,6))
    cont_btn.pack(side="right")

    opts = ttk.Frame(root, padding=(10,0))
    opts.pack(fill="x")
    ttk.Checkbutton(opts, text="Toujours au-dessus", variable=topmost_var, command=_toggle_topmost).pack(side="left")

    # --- Ajout UI: Mode récupération ---
    recover_var = tk.BooleanVar(value=False)
    last_x_var = tk.StringVar()
    last_y_var = tk.StringVar()
    hint_num_var = tk.IntVar(value=1)

    rec_frame = ttk.LabelFrame(root, text="Mode récupération", padding=10)
    rec_frame.pack(fill="x", padx=10, pady=(6,0))

    ttk.Checkbutton(rec_frame, text="Activer", variable=recover_var).grid(row=0, column=0, sticky="w", padx=(0,10))

    ttk.Label(rec_frame, text="X dernier indice:").grid(row=1, column=0, sticky="w")
    x_entry = ttk.Entry(rec_frame, textvariable=last_x_var, width=10)
    x_entry.grid(row=1, column=1, sticky="w", padx=(6,14))

    ttk.Label(rec_frame, text="Y dernier indice:").grid(row=1, column=2, sticky="w")
    y_entry = ttk.Entry(rec_frame, textvariable=last_y_var, width=10)
    y_entry.grid(row=1, column=3, sticky="w", padx=(6,14))

    ttk.Label(rec_frame, text="Indice à traiter (1..n):").grid(row=1, column=4, sticky="w")
    idx_entry = ttk.Entry(rec_frame, textvariable=hint_num_var, width=6)
    idx_entry.grid(row=1, column=5, sticky="w", padx=(6,0))

    def _toggle_rec_inputs(*_):
        state = "normal" if recover_var.get() else "disabled"
        for w in (x_entry, y_entry, idx_entry):
            w.configure(state=state)
    _toggle_rec_inputs()
    recover_var.trace_add("write", _toggle_rec_inputs)
    # --- fin ajout UI ---

    log_frame = ttk.Frame(root, padding=10)
    log_frame.pack(fill="both", expand=True)

    log_text = tk.Text(log_frame, height=24, wrap="word", state="disabled")
    log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
    log_text['yscrollcommand'] = log_scroll.set
    log_text.pack(side="left", fill="both", expand=True)
    log_scroll.pack(side="right", fill="y")

    # === AJOUT: Gestion des couleurs de fond selon l'état ===
    style = ttk.Style(root)

    # Applique un style de fond uniforme aux conteneurs et au texte
    def set_gui_status(state: str):
        # Couleurs: blanc (idle), vert (running), rouge (stopped)
        colors = {
            "idle":   "#FFFFFF",
            "running":"#CCFFCC",
            "stopped":"#FFCCCC",
        }
        color = colors.get(state, "#FFFFFF")
        root.configure(bg=color)
        # Styles ttk pour que les frames/labelframes héritent du fond
        style.configure("App.TFrame", background=color)
        style.configure("App.TLabelframe", background=color)
        style.configure("App.TLabelframe.Label", background=color)
        style.configure("App.TLabel", background=color)
        style.configure("App.TCheckbutton", background=color)
        style.configure("App.TButton", background=color)
        # Appliquer les styles aux conteneurs existants
        btn_frame.configure(style="App.TFrame")
        opts.configure(style="App.TFrame")
        rec_frame.configure(style="App.TLabelframe")
        log_frame.configure(style="App.TFrame")
        # Fond de la zone de logs
        try:
            log_text.configure(bg=color)
        except Exception:
            pass

    # État initial: blanc
    set_gui_status("idle")
    # === FIN AJOUT ===

    # Redirection des prints vers la zone de log (thread-safe via Queue)
    q = queue.Queue()
    writer = _QueueWriter(q)
    original_stdout = sys.stdout
    sys.stdout = writer

    def poll_log():
        try:
            while True:
                msg = q.get_nowait()
                # Détection d'ordres GUI spéciaux
                if "[GUI]ENABLE_CONTINUE" in msg:
                    cont_btn.configure(state="normal")
                    continue
                log_text.configure(state="normal")
                log_text.insert("end", msg)
                log_text.see("end")
                log_text.configure(state="disabled")
        except queue.Empty:
            pass
        root.after(100, poll_log)

    poll_log()

    # Message de démarrage (apparaît dans le journal GUI)
    show_startup_message("gui")

    t_holder = {"t": None}

    def start_bot():
        if t_holder["t"] and t_holder["t"].is_alive():
            return
        # Validation mode récupération
        start_pos_param = None
        start_hint_idx_param = None
        if recover_var.get():
            sx, sy = last_x_var.get().strip(), last_y_var.get().strip()
            if not sx or not sy:
                print("Mode récupération: veuillez renseigner X et Y.")
                return
            try:
                sidx = max(0, int(hint_num_var.get()) - 1)
            except Exception:
                print("Mode récupération: numéro d'indice invalide, utilisez 1..n.")
                return
            start_pos_param = (sx, sy)
            start_hint_idx_param = sidx

        STOP_EVENT.clear()
        RESUME_EVENT.set()
        start_btn.configure(state="disabled")
        stop_btn.configure(state="normal")
        cont_btn.configure(state="disabled")

        # Passe le fond au vert dès le lancement
        set_gui_status("running")

        def _run():
            try:
                load_config()
                get_ocr_instance()
                print("\nLe bot démarre (GUI)")
                if start_pos_param is not None or start_hint_idx_param is not None:
                    print("Mode récupération activé.")
                run_bot(stop_event=STOP_EVENT,
                        resume_event=RESUME_EVENT,
                        gui_mode=True,
                        start_pos=start_pos_param,
                        start_hint_index=start_hint_idx_param)
            except Exception as e:
                print(f"ERREUR: {e}")
            finally:
                # Quand le bot se termine ou est stoppé: fond rouge
                root.after(0, lambda: set_gui_status("stopped"))
                start_btn.configure(state="normal")
                stop_btn.configure(state="disabled")
                cont_btn.configure(state="disabled")
        t = threading.Thread(target=_run, daemon=True)
        t_holder["t"] = t
        t.start()

    def stop_bot():
        STOP_EVENT.set()
        RESUME_EVENT.set()
        # Passe immédiatement au rouge
        set_gui_status("stopped")
        print("Demande d'arrêt envoyée...")

    def do_calibration():
        if t_holder["t"] and t_holder["t"].is_alive():
            messagebox.showinfo("Info", "Arrêtez le bot avant de lancer la calibration.")
            return
        # Ouvre le gestionnaire de calibration (édition champ par champ)
        open_calibration_manager(root)

    def do_continue():
        RESUME_EVENT.set()
        cont_btn.configure(state="disabled")
        print("Reprise après Phorreur...")

    start_btn.configure(command=start_bot)
    stop_btn.configure(command=stop_bot)
    calib_btn.configure(command=do_calibration)
    cont_btn.configure(command=do_continue)

    def on_close():
        STOP_EVENT.set()
        RESUME_EVENT.set()
        sys.stdout = original_stdout
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    main()