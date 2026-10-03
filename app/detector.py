from pathlib import Path
import time

import cv2

from ultralytics import YOLO

from app.config import (
    MODEL_PATH,
    QUALITY_POLICY_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    IMAGE_SIZE,
    INFERENCE_DEVICE,
)

from ml.src.inference.quality_decision import (
    apply_quality_policy,
    get_bbox,
    load_quality_policy,
)


class DefectDetector:
    """
    Shared inference service used by the API.

    Responsibilities:
    - load the optimized YOLO model
    - run object detection
    - convert YOLO output into JSON-safe data
    - apply the Day 20 quality decision policy
    - draw annotated predictions
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        quality_policy_path: Path = (
            QUALITY_POLICY_PATH
        ),
    ):
        self.model_path = Path(
            model_path
        )

        self.quality_policy_path = Path(
            quality_policy_path
        )

        self.model = None

        self.quality_policy = (
            load_quality_policy(
                self.quality_policy_path
            )
        )

        self.load_model()

    # =====================================================
    # MODEL LOADING
    # =====================================================

    def load_model(
        self,
    ) -> None:
        """
        Load YOLO when the model artifact exists.

        Missing model weights do not crash
        the entire API. Health endpoints can
        still report model availability.
        """

        if not self.model_path.exists():

            print(
                "WARNING: optimized YOLO "
                "model not found."
            )

            print(
                f"Expected model: "
                f"{self.model_path}"
            )

            self.model = None

            return

        try:
            self.model = YOLO(
                str(
                    self.model_path
                )
            )

            print(
                "YOLO model loaded "
                "successfully."
            )

            print(
                f"Model path: "
                f"{self.model_path}"
            )

        except Exception as error:

            self.model = None

            print(
                f"Model loading error: "
                f"{error}"
            )

    # =====================================================
    # STATUS
    # =====================================================

    def is_ready(
        self,
    ) -> bool:
        return (
            self.model is not None
        )

    def get_classes(
        self,
    ):
        if not self.is_ready():
            return {}

        return self.model.names

    # =====================================================
    # VALIDATION
    # =====================================================

    @staticmethod
    def validate_confidence(
        confidence: float,
    ) -> float:
        confidence = float(
            confidence
        )

        if not (
            0.0
            <= confidence
            <= 1.0
        ):
            raise ValueError(
                "Confidence must be "
                "between 0 and 1."
            )

        return confidence

    # =====================================================
    # RESULT EXTRACTION
    # =====================================================

    def extract_detections(
        self,
        result,
    ) -> list[dict]:
        """
        Convert an Ultralytics result into
        simple JSON-compatible detections.
        """

        boxes = getattr(
            result,
            "boxes",
            None,
        )

        if boxes is None:
            return []

        xyxy_tensor = getattr(
            boxes,
            "xyxy",
            None,
        )

        confidence_tensor = getattr(
            boxes,
            "conf",
            None,
        )

        class_tensor = getattr(
            boxes,
            "cls",
            None,
        )

        if (
            xyxy_tensor is None
            or confidence_tensor is None
            or class_tensor is None
        ):
            return []

        xyxy_values = (
            xyxy_tensor
            .detach()
            .cpu()
            .numpy()
        )

        confidence_values = (
            confidence_tensor
            .detach()
            .cpu()
            .numpy()
        )

        class_values = (
            class_tensor
            .detach()
            .cpu()
            .numpy()
        )

        if not (
            len(xyxy_values)
            == len(confidence_values)
            == len(class_values)
        ):
            raise ValueError(
                "YOLO result tensors "
                "have inconsistent lengths."
            )

        names = getattr(
            result,
            "names",
            self.model.names,
        )

        detections = []

        for (
            box,
            confidence,
            class_value,
        ) in zip(
            xyxy_values,
            confidence_values,
            class_values,
        ):

            class_id = int(
                class_value
            )

            if isinstance(
                names,
                dict,
            ):
                class_name = names.get(
                    class_id,
                    str(class_id),
                )

            else:
                class_name = names[
                    class_id
                ]

            x1, y1, x2, y2 = [
                float(value)
                for value in box
            ]

            detections.append(
                {
                    "class_id":
                        class_id,

                    "class_name":
                        str(
                            class_name
                        ),

                    "confidence":
                        round(
                            float(
                                confidence
                            ),
                            4,
                        ),

                    "bbox": {
                        "x1":
                            round(
                                x1,
                                2,
                            ),

                        "y1":
                            round(
                                y1,
                                2,
                            ),

                        "x2":
                            round(
                                x2,
                                2,
                            ),

                        "y2":
                            round(
                                y2,
                                2,
                            ),
                    },
                }
            )

        return detections

    # =====================================================
    # RAW DETECTION
    # =====================================================

    def detect(
        self,
        image,
        confidence=None,
    ) -> dict:
        """
        Execute optimized YOLO inference.
        """

        if not self.is_ready():
            raise RuntimeError(
                "Optimized YOLO model "
                "is not loaded."
            )

        if confidence is None:
            confidence = (
                CONFIDENCE_THRESHOLD
            )

        confidence = (
            self.validate_confidence(
                confidence
            )
        )

        start_time = (
            time.perf_counter()
        )

        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device=INFERENCE_DEVICE,
            verbose=False,
        )

        inference_time_ms = (
            (
                time.perf_counter()
                - start_time
            )
            * 1000.0
        )

        if not results:
            detections = []

        else:
            detections = (
                self.extract_detections(
                    results[0]
                )
            )

        return {
            "detections":
                detections,

            "detection_count":
                len(
                    detections
                ),

            "inference_time_ms":
                round(
                    inference_time_ms,
                    2,
                ),
        }

    # =====================================================
    # QUALITY POLICY
    # =====================================================

    def apply_quality(
        self,
        image,
        detections: list[dict],
    ) -> dict:
        """
        Apply Day 20 operational triage
        to backend-format detections.
        """

        if image is None:
            raise ValueError(
                "Image cannot be None."
            )

        height, width = (
            image.shape[:2]
        )

        return apply_quality_policy(
            detections=detections,
            image_width=width,
            image_height=height,
            policy=self.quality_policy,
        )

    def detect_with_quality(
        self,
        image,
        confidence=None,
    ) -> dict:
        """
        Execute detection followed by
        PASS / REVIEW / REJECT processing.
        """

        detection_result = (
            self.detect(
                image=image,
                confidence=confidence,
            )
        )

        quality_result = (
            self.apply_quality(
                image=image,
                detections=(
                    detection_result[
                        "detections"
                    ]
                ),
            )
        )

        return {
            **detection_result,

            "quality":
                quality_result,
        }

    # =====================================================
    # ANNOTATED IMAGE
    # =====================================================

    def detect_and_draw(
        self,
        image,
        confidence=None,
    ):
        """
        Run the complete inference pipeline and
        draw the final post-processed detections.
        """

        result = (
            self.detect_with_quality(
                image=image,
                confidence=confidence,
            )
        )

        annotated_image = (
            image.copy()
        )

        detections = (
            result[
                "quality"
            ][
                "detections"
            ]
        )

        color_by_level = {
            "LOW":
                (
                    0,
                    180,
                    0,
                ),

            "MEDIUM":
                (
                    0,
                    180,
                    255,
                ),

            "HIGH":
                (
                    0,
                    0,
                    255,
                ),
        }

        for detection in detections:

            bbox = get_bbox(
                detection
            )

            x1 = int(
                bbox["x1"]
            )

            y1 = int(
                bbox["y1"]
            )

            x2 = int(
                bbox["x2"]
            )

            y2 = int(
                bbox["y2"]
            )

            class_name = (
                detection[
                    "class_name"
                ]
            )

            confidence_score = (
                detection[
                    "confidence"
                ]
            )

            triage_level = (
                detection[
                    "triage_level"
                ]
            )

            color = color_by_level.get(
                triage_level,
                (
                    255,
                    255,
                    255,
                ),
            )

            cv2.rectangle(
                annotated_image,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                color,
                2,
            )

            label = (
                f"{class_name} "
                f"{confidence_score:.2f} "
                f"[{triage_level}]"
            )

            cv2.putText(
                annotated_image,
                label,
                (
                    x1,
                    max(
                        20,
                        y1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        decision = (
            result[
                "quality"
            ][
                "decision"
            ]
        )

        cv2.putText(
            annotated_image,
            f"Decision: {decision}",
            (
                10,
                25,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                255,
                255,
                255,
            ),
            2,
        )

        return (
            annotated_image,
            result,
        )