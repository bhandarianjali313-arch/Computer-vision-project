import cv2
import numpy as np


# ==========================================
# IMAGE DECODING
# ==========================================

def decode_image(
    image_bytes
):

    try:

        array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )


        image = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR
        )


        if image is None:

            raise ValueError(
                "Unable to decode image."
            )


        return image


    except Exception as error:

        raise ValueError(
            f"Invalid image: {error}"
        )


# ==========================================
# IMAGE ENCODING
# ==========================================

def encode_jpeg(
    image,
    quality=90
):

    success, encoded = cv2.imencode(

        ".jpg",

        image,

        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality
        ]

    )


    if not success:

        raise ValueError(
            "Unable to encode image."
        )


    return encoded.tobytes()


# ==========================================
# IMAGE INFORMATION
# ==========================================

def get_image_info(
    image
):

    height, width = (
        image.shape[:2]
    )


    return {

        "width":
            width,

        "height":
            height,

        "channels":
            image.shape[2]
            if len(image.shape) == 3
            else 1
    }
