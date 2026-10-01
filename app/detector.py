import os
import time

import cv2
import numpy as np

from ultralytics import YOLO

from app.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    IMAGE_SIZE
)


class DefectDetector:

    def __init__(self):

        self.model = None

        self.load_model()


    # ==========================================
    # LOAD YOLO MODEL
    # ==========================================

    def load_model(self):

        if not os.path.exists(
            MODEL_PATH
        ):

            print(
                f"WARNING: YOLO model not found at "
                f"{MODEL_PATH}"
            )

            print(
                "Place your trained best.pt file "
                "inside the models folder."
            )

            return


        try:

            self.model = YOLO(
                MODEL_PATH
            )

            print(
                f"YOLO model loaded: "
                f"{MODEL_PATH}"
            )

        except Exception as error:

            print(
                f"Model loading error: {error}"
            )


    # ==========================================
    # MODEL STATUS
    # ==========================================

    def is_ready(self):

        return self.model is not None


    # ==========================================
    # GET CLASS NAMES
    # ==========================================

    def get_classes(self):

        if not self.is_ready():

            return {}

        return self.model.names


    # ==========================================
    # DETECT DEFECTS
    # ==========================================

    def detect(
        self,
        image,
        confidence=None
    ):

        if not self.is_ready():

            raise RuntimeError(
                "YOLO model is not loaded. "
                "Place models/best.pt in the project."
            )


        if confidence is None:

            confidence = (
                CONFIDENCE_THRESHOLD
            )


        start_time = time.perf_counter()


        # --------------------------------------
        # YOLO INFERENCE
        # --------------------------------------

        results = self.model.predict(

            source=image,

            conf=confidence,

            iou=IOU_THRESHOLD,

            imgsz=IMAGE_SIZE,

            verbose=False

        )


        inference_time = (
            time.perf_counter()
            - start_time
        )


        detections = []


        # --------------------------------------
        # PROCESS RESULTS
        # --------------------------------------

        for result in results:

            if result.boxes is None:

                continue


            boxes = (
                result.boxes
            )


            for i in range(
                len(boxes)
            ):

                box = boxes.xyxy[
                    i
                ].cpu().numpy()


                confidence_score = float(
                    boxes.conf[
                        i
                    ].cpu().item()
                )


                class_id = int(
                    boxes.cls[
                        i
                    ].cpu().item()
                )


                class_name = (
                    self.model.names[
                        class_id
                    ]
                )


                x1, y1, x2, y2 = (
                    map(
                        int,
                        box
                    )
                )


                detections.append({

                    "class_id":
                        class_id,

                    "class_name":
                        class_name,

                    "confidence":
                        round(
                            confidence_score,
                            4
                        ),

                    "bbox": {

                        "x1": x1,

                        "y1": y1,

                        "x2": x2,

                        "y2": y2
                    }
                })


        return {

            "detections":
                detections,

            "detection_count":
                len(detections),

            "inference_time_ms":
                round(
                    inference_time * 1000,
                    2
                )
        }


    # ==========================================
    # ANNOTATED IMAGE
    # ==========================================

    def detect_and_draw(
        self,
        image,
        confidence=None
    ):

        if not self.is_ready():

            raise RuntimeError(
                "YOLO model is not loaded."
            )


        if confidence is None:

            confidence = (
                CONFIDENCE_THRESHOLD
            )


        start_time = time.perf_counter()


        results = self.model.predict(

            source=image,

            conf=confidence,

            iou=IOU_THRESHOLD,

            imgsz=IMAGE_SIZE,

            verbose=False

        )


        inference_time = (
            time.perf_counter()
            - start_time
        )


        annotated_image = (
            image.copy()
        )

        detections = []


        for result in results:

            if result.boxes is None:

                continue


            boxes = result.boxes


            for i in range(
                len(boxes)
            ):

                box = boxes.xyxy[
                    i
                ].cpu().numpy()


                score = float(
                    boxes.conf[
                        i
                    ].cpu().item()
                )


                class_id = int(
                    boxes.cls[
                        i
                    ].cpu().item()
                )


                class_name = (
                    self.model.names[
                        class_id
                    ]
                )


                x1, y1, x2, y2 = map(
                    int,
                    box
                )


                # ----------------------------------
                # DRAW BOUNDING BOX
                # ----------------------------------

                cv2.rectangle(

                    annotated_image,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2

                )


                # ----------------------------------
                # LABEL
                # ----------------------------------

                label = (
                    f"{class_name} "
                    f"{score:.2f}"
                )


                cv2.putText(

                    annotated_image,

                    label,

                    (x1, max(y1 - 10, 20)),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (0, 255, 0),

                    2

                )


                detections.append({

                    "class_id":
                        class_id,

                    "class_name":
                        class_name,

                    "confidence":
                        round(
                            score,
                            4
                        ),

                    "bbox": {

                        "x1": x1,

                        "y1": y1,

                        "x2": x2,

                        "y2": y2
                    }
                })


        return (

            annotated_image,

            {

                "detections":
                    detections,

                "detection_count":
                    len(detections),

                "inference_time_ms":
                    round(
                        inference_time * 1000,
                        2
                    )
            }

        )
