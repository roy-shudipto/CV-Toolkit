# References:
# https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
# https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
# https://stackoverflow.com/questions/75581476/2d-to-3d-projection-opencv-gives-high-error
# https://stackoverflow.com/questions/18637494/camera-position-in-world-coordinate-from-cvsolvepnp?rq=1
# https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point

import cv2
import numpy as np
from loguru import logger

import defaults as d


def pnp_calc():
    # run OpenCV->solvePnP
    logger.info("Starting to run Opencv->solvePnP.")
    success, rotation_vector, translation_vector = cv2.solvePnP(
        np.array(d.WORLD_POINTS, dtype=np.float32).reshape((-1, 3)),
        np.array(d.IMAGE_POINTS, dtype=np.float32).reshape((-1, 2)),
        np.array(d.CAMERA_MATRIX, dtype=np.float32),
        np.array(d.DISTORTION_COEFFICIENT, dtype=np.float32),
    )
    if success is True:
        logger.info("Successfully ran Opencv->solvePnP.")
    else:
        logger.error("Failed to run Opencv->solvePnP.")
        exit(1)

    # test: world-point to image-point
    logger.info(
        "Starting to run calculation and validation for world-point to image-point."
    )
    (image_point_calc, _) = cv2.projectPoints(
        np.array([d.WORLD_POINT_VAL]),
        rotation_vector,
        translation_vector,
        np.array(d.CAMERA_MATRIX, dtype=np.float32),
        np.array(d.DISTORTION_COEFFICIENT, dtype=np.float32),
    )
    logger.info(f"Calculated image-point: {image_point_calc[0][0].astype(int)}")
    logger.info(f"Ground-truth image-point: {d.IMAGE_POINT_VAL}")

    # test: image-point to world-point
    logger.info(
        "Starting to run calculation and validation for image-point to world-point."
    )
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)  # 3x3
    translation_mat = translation_vector.reshape((3, 1))  # 3x1
    camera_mat = np.array(d.CAMERA_MATRIX, dtype=np.float32)  # 3x3
    image_point_mat = np.array(d.IMAGE_POINT_VAL + [1], dtype=np.float32).reshape(
        (3, 1)
    )  # 3x1

    # reference: https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
    left_side_mat = np.matmul(
        np.matmul(np.linalg.inv(rotation_mat), np.linalg.inv(camera_mat)),
        image_point_mat,
    )  # 3x1

    right_side_mat = np.matmul(np.linalg.inv(rotation_mat), translation_mat)  # 3x1
    scale = (d.WORLD_POINT_VAL[-1] + right_side_mat[2, 0]) / left_side_mat[2, 0]
    world_point_mat = np.matmul(
        np.linalg.inv(rotation_mat),
        (
            np.matmul(scale * np.linalg.inv(camera_mat), image_point_mat)
            - translation_mat
        ),
    )  # 3x1
    world_point_mat = np.round(world_point_mat.flatten(), decimals=1)
    logger.info(f"Calculated world-point: {world_point_mat.tolist()}")
    logger.info(f"Ground-truth world-point: {d.WORLD_POINT_VAL}")


if __name__ == "__main__":
    pnp_calc()
