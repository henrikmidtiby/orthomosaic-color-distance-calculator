import numpy as np
import cv2
import rasterio
import random
from rasterio.windows import Window
from skimage.segmentation import slic
from rasterio.transform import Affine
import time
from icecream import ic
import argparse
from tqdm import tqdm

# Initial code from Elzbieta Pastucha for counting pumpkins.


class Tile:
    def __init__(self, start_point, position, height, width):
        self.size = (height, width)
        self.tile_position = position
        self.ulc = start_point
        self.ulc_global = []
        self.lrc = (start_point[0] + height, start_point[1] + width)
        self.processing_range = [[0, 0], [0, 0]]


def rasterio_opencv2(image):
    if image.shape[0] >= 3:  # might include alpha channel
        false_color_img = image.transpose(1, 2, 0)
        separate_colors = cv2.split(false_color_img)
        return cv2.merge([separate_colors[2], separate_colors[1], separate_colors[0]])
    else:
        return(image)


def read_tile(orthomosaic, tile):
    with rasterio.open(orthomosaic) as src:
        im = src.read(window=Window.from_slices((tile.ulc[0], tile.lrc[0]),
                                                (tile.ulc[1], tile.lrc[1])))
    return rasterio_opencv2(im)


class ColorModel():
    """
    A multivariate normal distribution used to describe the color of a set of pixels.
    """
    def __init__(self):
        self.average = None
        self.covariance = None
        self.reference_image = None

    def load_reference_image(self, filename_reference_image):
        self.reference_image = cv2.imread(filename_reference_image)

    def load_annotated_image(self, filename_annotated_image):
        self.annotated_image = cv2.imread(filename_annotated_image)

    def generate_pixel_mask(self, lower_range = (0, 0, 245), higher_range = (10, 10, 256)):
        self.pixel_mask = cv2.inRange(self.annotated_image, lower_range, higher_range)

    def calculate_statistics(self):
        pixels = np.reshape(self.reference_image, (-1, 3))
        mask_pixels = np.reshape(self.pixel_mask, (-1))

        # Using numpy to calculate mean and covariance matrix
        self.covariance = np.cov(pixels.transpose(), aweights=mask_pixels)
        self.average = np.average(pixels.transpose(), weights=mask_pixels, axis=1)

    def show_statistics(self):
        print("Average color value of annotated pixels")
        print(self.average)
        print("Covariance matrix of the annotated pixels")
        print(self.covariance)



# initial setup
orthos = ['/home/hemi/Nextcloud/Shared/2023/2023-02-14 ScoutRobotics data/Orthomosaic-møn-mark3.tif']


class ColorBasedSegmenter():
    def __init__(self):
        self.tile_size = 3000
        self.colormodel = ColorModel()
        self.ref_image_filename = None
        self.ref_image_annotated_filename = None
        self.output_scale_factor = None
        self.mahal_tile_location = None
        self.input_tile_location = None


    def main(self, filename_orthomosaic):
        self.initialize_color_model(self.ref_image_filename, self.ref_image_annotated_filename)
        self.process_orthomosaic(filename_orthomosaic)


    def initialize_color_model(self, ref_image_filename, ref_image_annotated_filename):
        self.colormodel.load_reference_image(ref_image_filename)
        self.colormodel.load_annotated_image(ref_image_annotated_filename)
        self.colormodel.generate_pixel_mask()
        self.colormodel.calculate_statistics()
        self.colormodel.show_statistics()


    def process_orthomosaic(self, filename_orthomosaic):
        start_time = time.time()
        self.calculate_color_distances_in_orthomosaic(filename_orthomosaic)
        proc_time = time.time() - start_time
        print('Calculation of color distances: ', proc_time)


    def define_tiles(self, filename_orthomosaic, overlap, height, width):
        """
        Given a path to an orthomosaic, create a list of tiles which covers the
        orthomosaic with a specified overlap, height and width.
        """

        with rasterio.open(filename_orthomosaic) as src:
            columns = src.width
            rows = src.height

        last_position = (rows - height, columns - width)

        n_height = np.ceil(rows / (height * (1 - overlap))).astype(int)
        n_width = np.ceil(columns / (width * (1 - overlap))).astype(int)

        step_height = np.trunc(last_position[0] / (n_height - 1)).astype(int)
        step_width = np.trunc(last_position[1] / (n_width - 1)).astype(int)

        tiles = []
        for r in range(0, n_height):
            for c in range(0, n_width):
                pos = [r, c]
                if r == (n_height - 1):
                    tile_r = last_position[0]
                else:
                    tile_r = r * step_height
                if c == (n_width - 1):
                    tile_c = last_position[1]
                else:
                    tile_c = c * step_width
                tiles.append(Tile((tile_r, tile_c), pos, height, width))

        return tiles, step_width, step_height


    def calculate_mahalanobis_distance(self, image):
        """
        For all pixels in the image, calculate the Mahalanobis distance 
        to the reference color.
        """
        pixels = np.reshape(image, (-1, 3))
        inv_cov = np.linalg.inv(self.colormodel.covariance)
        diff = pixels - self.colormodel.average
        moddotproduct = diff * (diff @ inv_cov)
        mahalanobis_dist = np.sum(moddotproduct, axis=1)
        mahalanobis_dist = np.sqrt(mahalanobis_dist)
        mahalanobis_distance_image_in_function = np.reshape(mahalanobis_dist, (image.shape[0], image.shape[1]))

        return mahalanobis_distance_image_in_function


    def calculate_color_distances_in_orthomosaic(self, filename_orthomosaic):
        """
        For all pixels in the orthomosaic, calculate the Mahalanobis distance 
        to the reference color.
        """
        with rasterio.open(filename_orthomosaic) as src:
            self.resolution = src.res
            self.crs = src.crs
            self.left = src.bounds[0]
            self.top = src.bounds[3]

        processing_tiles = self.get_processing_tiles(filename_orthomosaic, self.tile_size)

        for tile_number, tile in enumerate(tqdm(processing_tiles)):
            img_RGB = read_tile(filename_orthomosaic, tile)
            if self.is_image_empty(img_RGB):
                continue

            self.process_tile(filename_orthomosaic, img_RGB, tile_number, tile)


    def get_processing_tiles(self, filename_orthomosaic, tile_size):
        """
        Generate a list of tiles to process, including a padding region around the actual tile.
        Takes care of edge cases, where the tile does not have adjacent tiles in all directions.
        """
        processing_tiles, st_width, st_height = self.define_tiles(filename_orthomosaic, 0.01, tile_size, tile_size)

        no_r = np.max([t.tile_position[0] for t in processing_tiles])
        no_c = np.max([t.tile_position[1] for t in processing_tiles])

        half_overlap_c = (tile_size-st_width)/2
        half_overlap_r = (tile_size-st_height)/2

        for tile in processing_tiles:
            tile.processing_range = [[half_overlap_r, tile_size - half_overlap_r],
                                     [half_overlap_c, tile_size - half_overlap_c]]
            if tile.tile_position[0] == 0:
                tile.processing_range[0][0] = 0
            if tile.tile_position[0] == no_r:
                tile.processing_range[0][1] = tile_size
            if tile.tile_position[1] == 0:
                tile.processing_range[0][0] = 0
            if tile.tile_position[1] == no_c:
                tile.processing_range[0][1] = tile_size

        return processing_tiles


    def is_image_empty(self, image):
        """Helper function for deciding if an image contains no data."""
        return np.max(image[:, :, 0]) == np.min(image[:, :, 0])


    def process_tile(self, filename_orthomosaic, img_RGB, tile_number, tile):
        tile.ulc_global = [
                self.top - (tile.ulc[0] * self.resolution[0]), 
                self.left + (tile.ulc[1] * self.resolution[1])]

        mahalanobis_distance_image = self.calculate_mahalanobis_distance(img_RGB[:, :, :])
        mahal = cv2.convertScaleAbs(mahalanobis_distance_image, alpha=self.output_scale_factor, beta = 0)
        mahal = mahal.astype(np.uint8)

        width = tile.size[1]
        height = tile.size[0]

        transform = Affine.translation(tile.ulc_global[1] + self.resolution[0] / 2, 
                                       tile.ulc_global[0] - self.resolution[0] / 2) * \
                    Affine.scale(self.resolution[0], -self.resolution[0])

        # optional save of results - just lob detection and thresholding result
        self.save_results(img_RGB, tile_number, mahal, filename_orthomosaic, self.resolution, height, width, self.crs, transform)



    def save_results(self, img_RGB, tile_number, mahal, filename_orthomosaic, res, height, width, crs, transform):
        if self.input_tile_location is not None:
            name_annotated_image = f'{ self.input_tile_location }{ tile_number:04d}.tiff'
            img_to_save = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2RGB)
            temp_to_save = img_to_save.transpose(2, 0, 1)
            new_dataset = rasterio.open(name_annotated_image,
                                        'w',
                                        driver='GTiff',
                                        res=res,
                                        height=height,
                                        width=width,
                                        count=3,
                                        dtype=temp_to_save.dtype,
                                        crs=crs,
                                        transform=transform)
            new_dataset.write(temp_to_save)
            new_dataset.close()

        if self.mahal_tile_location is not None:
            name_mahal_results = f'{ self.mahal_tile_location }{ tile_number:04d}.tiff'
            img_to_save = mahal
            temp_to_save = img_to_save.reshape(1, img_to_save.shape[0], img_to_save.shape[1])
            # The ordering should be color channels, width and height.
            new_dataset = rasterio.open(name_mahal_results,
                                        'w',
                                        driver='GTiff',
                                        res=res,
                                        height=height,
                                        width=width,
                                        count=1,
                                        dtype=temp_to_save.dtype,
                                        crs=crs,
                                        transform=transform)
            new_dataset.write(temp_to_save)
            new_dataset.close()


parser = argparse.ArgumentParser(
          prog = 'ColorBasedSegmeneter', 
          description = 'A tool for calculating color distances in an orthomosaic to a reference color based on samples from an annotated image')
parser.add_argument('orthomosaic', 
                    help = 'Path to the orthomosaic that you want to process')
parser.add_argument('reference', 
                    help = 'Path to the reference image')
parser.add_argument('annotated', 
                    help = 'Path to the annotated reference image')
parser.add_argument('--scale', 
                    default = 5, 
                    type = float, 
                    help = 'The calculated distances are multiplied with this factor before the result is saved as an image. Default value is 5.')
parser.add_argument('--tilesize', 
                    default = 3000, 
                    help = 'The height and width of tiles that are analyzed. Default is 3000.')
parser.add_argument('--mahal_tile_location', 
                    default = 'output/mahal', 
                    help = 'The location in which to save the mahalanobis tiles.')
parser.add_argument('--input_tile_location', 
                    default = None, 
                    help = 'The location in which to save the input tiles.')
args = parser.parse_args()


cbs = ColorBasedSegmenter()
cbs.ref_image_filename = args.reference
cbs.ref_image_annotated_filename = args.annotated
cbs.output_scale_factor = args.scale
cbs.tile_size = args.tilesize
cbs.mahal_tile_location = args.mahal_tile_location
cbs.input_tile_location = args.input_tile_location
cbs.main(args.orthomosaic)
