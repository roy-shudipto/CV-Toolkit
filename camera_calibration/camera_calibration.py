# References:
# https://learnopencv.com/camera-calibration-using-opencv/
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# https://theailearner.com/tag/cv2-cornersubpix/

import click
import cv2
import numpy as np
import pathlib
from loguru import logger
from typing import Tuple


@click.command()
@click.option("--images", required=True, type=str, help="Path to the image directory.")
@click.option(
    "--checkerboard",
    required=True,
    nargs=2,
    type=click.Tuple([int, int]),
    help="Dimension of the checkerboard grid.",
)
@click.option(
    "--refine_window",
    required=False,
    default=(5, 5),
    nargs=2,
    type=click.Tuple([int, int]),
    help="Window size to refine detected corners.",
)
@click.option(
    "--debug", is_flag=True, help="Flag to pause on each image while calibration."
)
def calibrate_camera(
    images: str,
    checkerboard: Tuple[int, int],
    refine_window: Tuple[int, int],
    debug: bool,
) -> None:
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0 : checkerboard[0], 0 : checkerboard[1]].T.reshape(-1, 2)

    # for each image get image-points and corresponding object-point
    image_width, image_height = (None, None)
    cv2.namedWindow("Camera Calibration", cv2.WINDOW_AUTOSIZE)
    for image_file in pathlib.Path(images).iterdir():
        # read image
        image = cv2.imread(image_file.as_posix(), 1)

        # get size
        try:
            image_height, image_width, _ = image.shape
        except AttributeError:
            logger.error(f"Failed to read image: {image_file}")
            continue

        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find the chess board corners. If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
            gray,
            checkerboard,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if ret is True:
            # refine corner pixel coordinates for given 2d points.
            # refine_window: Size of the neighborhood where it searches for corners.
            # This is the half of the side length of the search window.
            # For example, if refine_window=Size(5,5) , then a (5∗2+1)×(5∗2+1)=11×11 search window is used.
            corners_refined = cv2.cornerSubPix(
                gray, corners, refine_window, (-1, -1), criteria
            )

            # add image points and object points
            img_points.append(corners_refined)
            obj_points.append(objp)

            # draw and display the refined corners
            image_disp = cv2.drawChessboardCorners(
                image, checkerboard, corners_refined, ret
            )
            cv2.imshow("Camera Calibration", image_disp)
            cv2.waitKey(0) if debug is True else cv2.waitKey(500)

    # destroy displayed window
    cv2.destroyAllWindows()

    # perform camera calibration by passing the value of known 3D points (obj_points) and,
    # corresponding pixel coordinates of the detected corners (img_points)
    ret, camera_matrix, dist_coefficient, _, _ = cv2.calibrateCamera(
        obj_points, img_points, (image_width, image_height), None, None
    )

    # print results
    logger.info(f"Camera Matrix: {camera_matrix.tolist()}")
    logger.info(f"Distortion Coefficient: {dist_coefficient.tolist()}")


if __name__ == "__main__":
    logger.info("Running camera calibration.")
    calibrate_camera()
