from facenet_pytorch import MTCNN, fixed_image_standardization
from torchvision import datasets, transforms
import numpy as np


class MTCNNDatasetWrapper(datasets.ImageFolder):
    def __init__(self, root_folder_path: str, device="cuda", return_boxes=True):
        super().__init__(root_folder_path)
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
            device=device
        )
        self.return_boxes = return_boxes
        self.resize_mtcnn = transforms.Compose([
            transforms.Resize((160, 160))])
        self.uncropped_transformations = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])
        self.cropped_transformations = transforms.Compose([
            fixed_image_standardization
        ])

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        image = self.resize_mtcnn(image)
        if self.return_boxes:
            box, _ = self.mtcnn.detect(image, landmarks=False)
            return image, label, box
        else:
            image_cropped = self.mtcnn(image)
            if image_cropped is not None:
                img_tensor = self.cropped_transformations(image_cropped)
            else:
                img_tensor = self.uncropped_transformations(image)
            return img_tensor, label
