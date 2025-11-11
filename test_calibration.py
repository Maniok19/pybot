#!/usr/bin/env python3
"""Test script pour le système de calibration amélioré."""

import json
import os
import sys

test_config = {
    "open_dofus": {"x": 930, "y": 887},
    "open_chrome": {"x": 751, "y": 889},
    "open_chasse_tab": {"x": 419, "y": 444},
    "dofus_chat_input": {"x": 242, "y": 839},
    "starting_pos_region": {"x": 71, "y": 255, "w": 51, "h": 19},
    "current_pos_region": {"x": 0, "y": 86, "w": 85, "h": 22},
    "input_pos_x": {"x": 652, "y": 291},
    "input_pos_y": {"x": 929, "y": 292},
    "dofus_ui_deadpoint": {"x": 1195, "y": 780},
    "chasse_deadpoint": {"x": 365, "y": 505},
    "chasse_direction_buttons": {
        "nord": {"x": 789, "y": 389},
        "est": {"x": 830, "y": 427},
        "ouest": {"x": 750, "y": 429},
        "sud": {"x": 790, "y": 462}
    },
    "chasse_element_input": {"x": 650, "y": 551},
    "chasse_element_option": {"x": 579, "y": 588},
    "hint_hover_first": {"x": 93, "y": 308},
    "hint_region_first": {"x": 252, "y": 284, "w": 322, "h": 71},
    "hint_click_first": {"x": 226, "y": 308},
    "row_step": 23,
    "travel_confirm_button": {"x": 729, "y": 490},
    "validate_click_offset_y": -4
}

class MockCalibrationManager:
    """Version simplifiée pour tester la validation."""
    
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
                print(f"❌ Clé point manquante ou invalide: {key}")
                return False
            if "x" not in cfg[key] or "y" not in cfg[key]:
                print(f"❌ Coordonnées x/y manquantes pour: {key}")
                return False
        
        for key in required_region_keys:
            if key not in cfg or not isinstance(cfg[key], dict):
                print(f"❌ Clé région manquante ou invalide: {key}")
                return False
            if not all(k in cfg[key] for k in ["x", "y", "w", "h"]):
                print(f"❌ Coordonnées x/y/w/h manquantes pour: {key}")
                return False
        
        if "chasse_direction_buttons" not in cfg:
            print(f"❌ Clé chasse_direction_buttons manquante")
            return False
        for dir_key in ["nord", "est", "ouest", "sud"]:
            if dir_key not in cfg["chasse_direction_buttons"]:
                print(f"❌ Direction manquante: {dir_key}")
                return False
            btn = cfg["chasse_direction_buttons"][dir_key]
            if "x" not in btn or "y" not in btn:
                print(f"❌ Coordonnées x/y manquantes pour direction: {dir_key}")
                return False
        
        if "row_step" not in cfg or not isinstance(cfg["row_step"], (int, float)):
            print(f"❌ row_step manquant ou invalide")
            return False
        
        return True

def test_validation():
    """Test la validation de configuration."""
    print("=== Test de validation ===")
    manager = MockCalibrationManager()
    
    # Test 1: Configuration valide
    print("\n1. Test configuration valide...")
    if manager.validate_config(test_config):
        print("✅ Configuration valide détectée correctement")
    else:
        print("❌ ÉCHEC: Configuration valide rejetée")
        return False
    
    # Test 2: Configuration avec clé manquante
    print("\n2. Test configuration avec clé manquante...")
    invalid_cfg = test_config.copy()
    del invalid_cfg["open_dofus"]
    if not manager.validate_config(invalid_cfg):
        print("✅ Configuration invalide détectée correctement")
    else:
        print("❌ ÉCHEC: Configuration invalide acceptée")
        return False
    
    # Test 3: Configuration avec région incomplète
    print("\n3. Test configuration avec région incomplète...")
    invalid_cfg = test_config.copy()
    invalid_cfg["starting_pos_region"] = {"x": 10, "y": 20}  # manque w et h
    if not manager.validate_config(invalid_cfg):
        print("✅ Région invalide détectée correctement")
    else:
        print("❌ ÉCHEC: Région invalide acceptée")
        return False
    
    # Test 4: Configuration avec direction manquante
    print("\n4. Test configuration avec direction manquante...")
    invalid_cfg = test_config.copy()
    invalid_cfg["chasse_direction_buttons"] = {
        "nord": {"x": 100, "y": 200},
        "est": {"x": 100, "y": 200}
        # manque ouest et sud
    }
    if not manager.validate_config(invalid_cfg):
        print("✅ Directions incomplètes détectées correctement")
    else:
        print("❌ ÉCHEC: Directions incomplètes acceptées")
        return False
    
    return True

def test_json_operations():
    """Test les opérations de lecture/écriture JSON."""
    print("\n=== Test opérations JSON ===")
    test_file = "/tmp/test_calib.json"
    
    # Test 1: Écriture
    print("\n1. Test écriture JSON...")
    try:
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_config, f, indent=2, ensure_ascii=False)
        print("✅ Écriture JSON réussie")
    except Exception as e:
        print(f"❌ ÉCHEC écriture: {e}")
        return False
    
    # Test 2: Lecture
    print("\n2. Test lecture JSON...")
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if loaded == test_config:
            print("✅ Lecture JSON réussie et données intègres")
        else:
            print("❌ ÉCHEC: Données lues différentes")
            return False
    except Exception as e:
        print(f"❌ ÉCHEC lecture: {e}")
        return False
    
    # Test 3: Backup
    print("\n3. Test backup...")
    backup_file = test_file + ".bak"
    try:
        import shutil
        shutil.copy2(test_file, backup_file)
        if os.path.exists(backup_file):
            print("✅ Backup créé avec succès")
        else:
            print("❌ ÉCHEC: Backup non créé")
            return False
    except Exception as e:
        print(f"❌ ÉCHEC backup: {e}")
        return False
    
    # Nettoyage
    try:
        os.remove(test_file)
        os.remove(backup_file)
    except Exception:
        pass
    
    return True

def main():
    """Exécute tous les tests."""
    print("╔════════════════════════════════════════════╗")
    print("║  Test du système de calibration amélioré  ║")
    print("╚════════════════════════════════════════════╝\n")
    
    all_pass = True
    
    if not test_validation():
        all_pass = False
    
    if not test_json_operations():
        all_pass = False
    
    print("\n" + "="*50)
    if all_pass:
        print("✅ TOUS LES TESTS SONT PASSÉS")
        print("="*50)
        return 0
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("="*50)
        return 1

if __name__ == "__main__":
    sys.exit(main())
