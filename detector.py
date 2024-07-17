import json
from functools import partial
from detectors.CargoLabelDet import CargoLabelDetector
from detectors.PalletCornerDet import PalletDetector
from detectors.SealDet import SealDetector
from detectors.ChaituoDet import ChaituoDetector
from detectors.FodsDet import FodsDetector
from detectors.QianhouDet import QianhouDetector
from containumber.manager import Manager
from detectors.ZheDangDet import ZheDangDetector
from detectors.StrapDet import StrapDetector


class Detector:
    def __init__(self):
        with open('detectors/detectors.json', 'r') as file:
            self.detectors_config = json.load(file)

        self.detector_instances = {
            'CargoLabel': CargoLabelDetector(),
            'PalletCorner': PalletDetector(),
            'Zhedang': ZheDangDetector(),
            'Strap': StrapDetector(),
            'Seal': SealDetector(),
            'Chaituo': ChaituoDetector(),
            'Fods': FodsDetector(),
            'Qianhou': QianhouDetector(),
            'ContainerNumber': Manager()
        }

        self._load_detector_models()

    def _load_detector_models(self):
        for name, instance in self.detector_instances.items():
            if name in self.detectors_config:
                model_path = self.detectors_config[name]['model_path']
                instance.load_model(model_path)

    def detect_all(self, img, task):
        task_mapping = {
            '1': 'cargo_label',
            '2': 'pallet_corner',
            '3': 'strap',
            '4': 'seal',
            '5': 'container',
            '6': 'chaituo',
            '7': 'fods',
            '8': 'qianhou',
            '9': 'zhedang'
        }

        if task in task_mapping:
            method_name = f"detect_{task_mapping[task]}"
            return getattr(self, method_name)(img)
        else:
            raise ValueError(f"Unknown task: {task}")

    def detect_cargo_label(self, img):
        return self.detector_instances['CargoLabel'].detect_cargo_label(img)

    def detect_zhedang(self, img):
        return self.detector_instances['Zhedang'].detect_zhedang_label(img)

    def detect_pallet_corner(self, img):
        return self.detector_instances['PalletCorner'].detect_pallet(img)

    def detect_strap(self, img):
        return self.detector_instances['Strap'].detect_strap(img)

    def detect_seal(self, img):
        return self.detector_instances['Seal'].detect_seal(img)

    def detect_container(self, img):
        return self.detector_instances['ContainerNumber'].detect_container_number(img)

    def detect_chaituo(self, img):
        return self.detector_instances['Chaituo'].detect_chaituo(img)

    def detect_fods(self, img):
        return self.detector_instances['Fods'].detect_fods(img)

    def detect_qianhou(self, img):
        return self.detector_instances['Qianhou'].detect_qianhou(img)