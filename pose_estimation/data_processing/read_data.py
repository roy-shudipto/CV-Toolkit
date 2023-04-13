import click
import pathlib
import yaml
from loguru import logger


@click.command()
@click.option("--yaml_file", required=True, type=str, help="Path to a .yaml file.")
def read_data(yaml_file):
    if pathlib.Path(yaml_file).suffix.lower() != ".yaml":
        logger.error(f"Label-file: {yaml_file} does not have [.yaml] extension.")
        exit(1)

    # read .yaml file
    data_dict = yaml.safe_load(open(yaml_file, "r"))

    # read image-points and world-points
    image_points = []
    world_points = []
    for point_name, point_dict in data_dict.items():
        try:
            image_points.append(point_dict["image_point"])
            world_points.append(point_dict["world_point"])
        except KeyError:
            logger.error(f"Invalid data: {point_name}: {point_dict}.")
            exit(1)

    # print image-points and world-points
    logger.info(f"Image-points: {image_points}")
    logger.info(f"World-points: {world_points}")


if __name__ == "__main__":
    logger.info("Reading data for pose_estimation.")
    read_data()
