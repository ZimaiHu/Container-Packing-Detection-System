import json
from detectors.CargoLabelDet import CargoLabelDetector
from detectors.PalletCornerDet import PalletDetector
from detectors.StrapDet import StrapDetector
from detectors.SealDet import SealDetector
from detectors.ContainerNumberDet import ContainerDetector
from detectors.ChaituoDet import ChaituoDetector
from detectors.FodsDet import FodsDetector
with open('detectors/detectors.json', 'r') as file:
    detectors = json.load(file)
class Detector:
    def __init__(self):
        self.PalletCorner = PalletDetector()
        self.CargoLabel = CargoLabelDetector()
        self.StrapDet= StrapDetector()
        self.SealDet = SealDetector()
        self.ContainerNumberDet = ContainerDetector()
        self.ChaituoDet = ChaituoDetector()
        self.FodsDet = FodsDetector()
        # 初始化检测器模型
        self._load_detector_model()
    def _load_detector_model(self):
        PalletCornerPath=detectors["PalletCorner"]["model_path"]
        CargoLabelPath = detectors["CargoLabel"]["model_path"]
        SealDetPath = detectors["Seal"]["model_path"]
        FodsDetPath = detectors["Fod"]["model_path"]
        self.CargoLabel.load_model(CargoLabelPath)
        self.PalletCorner.load_model(PalletCornerPath)
        self.SealDet.load_model(SealDetPath)
        self.ContainerNumberDet.load_model()
        self.ChaituoDet.load_model()
        self.FodsDet.load_model(FodsDetPath)
    # 通用检测器：通过检测任务调用
    def detect_all(self, img,task):
        if task == "1":
            return self.detect_cargo_label_det(img)
        if task == "2":
            return self.detect_pallet_corner_det(img)
        if task == "3":
            return self.detect_strap_det(img)
        if task == "4":
            return self.detect_seal_det(img)
        if task == "5":
            return self.detect_container_det(img)
        if task == "6":
            return self.detect_chaituo_det(img)
        if task == "7":
            return self.detect_fods_det(img)
    def detect_cargo_label_det(self, img):
        txt = self.CargoLabel.detect_cargo_label(img)
        return txt
    def detect_pallet_corner_det(self, img):
        txt = self.PalletCorner.detect_pallet(img)
        return txt
    def detect_strap_det(self, img):
        txt = self.StrapDet.detect_strap(img)
        return txt
    def detect_seal_det(self, img):
        txt = self.SealDet.detect_seal(img)
        return txt
    def detect_container_det(self, img):
        txt = self.ContainerNumberDet.detect_container(img)
        return txt
    def detect_chaituo_det(self, img):
        txt = self.ChaituoDet.detect_chaituo(img)
        return txt
    def detect_fods_det(self, img):
        txt= self.FodsDet.detect_fods(img)
        return txt