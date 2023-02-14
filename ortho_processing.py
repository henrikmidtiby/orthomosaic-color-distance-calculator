import numpy as np
import cv2 as cv
import rasterio
import random
from rasterio.windows import Window
from skimage.segmentation import slic
from rasterio.transform import Affine
import time
from icecream import ic

# Initial code from Elzbieta Pastucha for counting pumpkins.


class Tile:
    def __init__(self, start_point, position, height, width):
        self.size = (height, width)
        self.tile_position = position
        self.ulc = start_point
        self.ulc_global = []
        self.lrc = (start_point[0] + height, start_point[1] + width)
        self.rows_range = width
        self.columns_range = height
        self.pumpkin_list = []
        self.pumpkins_global = []
        self.contours = []
        self.processing_range = [[0, 0], [0, 0]]


def rasterio_opencv(image):
    if image.shape[0] >= 3:  # might include alpha channel
        false_color_img = image.transpose(1, 2, 0)
        separate_colors = cv.split(false_color_img)
        return cv.merge([separate_colors[2], separate_colors[1], separate_colors[0]])
    else:
        return(image)


def read_tile(orthomosaic, tile):
    with rasterio.open(orthomosaic) as src:
        im = src.read(window=Window.from_slices((tile.ulc[0], tile.lrc[0]),
                                                (tile.ulc[1], tile.lrc[1])))
    return rasterio_opencv(im)



class ExtractSaturatedPixelSegments():
    def __init__(self):
        self.pixels = []
        self.segmented = {}
        self.inliers = {}

    def main(self, orthomosaic, tiles, tiles_to_process, segmentation_parameters):
        for number in tiles_to_process:
            tile = tiles[number]
            tile_im = read_tile(orthomosaic, tile)
            image_thresholded = np.zeros_like(tile_im[:, :, 0])
            img_HLS = cv.cvtColor(tile_im, cv.COLOR_BGR2HLS)

            tile_segments = slic(img_HLS[:, :, 2],
                                 n_segments=segmentation_parameters[0],
                                 compactness=segmentation_parameters[1])
            self.segmented[str(number)] = tile_segments

            threshold = np.median(img_HLS[:, :, 2])
            std_image = np.std(img_HLS[:, :, 2])
            seg_in = []

            for i in range(0, np.max(tile_segments) + 1):
                mask = np.zeros_like(tile_segments)
                mask[tile_segments != i] = 1
                img_temp = img_HLS.copy().astype(float)
                img_temp[mask > 0] = float('nan')

                to_compare = [np.nanmean(img_temp[:, :, 2]), np.nanstd(img_temp[:, :, 2])]
                local_value = to_compare[0] - segmentation_parameters[2] * std_image

                if local_value > threshold:
                    seg_in.append(i)
                    image_thresholded[mask == 0] = 1
                    img_flat = np.reshape([img_temp], (-1, 3))
                    object_pixels = np.delete(img_flat, np.where(np.reshape(mask, -1) > 0), axis=0)
                    self.pixels.append(object_pixels)
            self.inliers[str(number)] = seg_in

        for i in range(0, len(self.pixels)):
            if i == 0:
                pixels_all = self.pixels[i]
            else:
                pixels_all = np.concatenate((pixels_all, self.pixels[i]), axis=0)

        return pixels_all


    






# initial setup
orthos = ['/home/hemi/Nextcloud/Shared/2023/2023-02-14 ScoutRobotics data/Orthomosaic-møn-mark3.tif']


class PumpkinCounter():
    def __init__(self):
        self.tile_size = 3000
        self.reference_color = None
        self.reference_color_covariance = None
        self.thr_mahalanobis = None


    def main(self, orthos):
        ref_image_filename = "data/crop_from_orthomosaic.png"
        ref_image_annotated_filename = "data/crop_from_orthomosaic_annotated.png"

        self.ref_image = cv2.imread(ref_image_filename)
        self.ref_image_annotated = cv2.imread(ref_image_annotated_filename)

        self.determine_colormodel_from_annotated_image(self.ref_image, self.ref_image_annotated)
        return 

        for ortho in orthos:
            self.process_orthomosaic(ortho)

    def determine_colormodel_from_annotated_image(self, ref_image, ref_image_annotated):
        pass
      
      


    def process_orthomosaic(self, ortho):
        start_time = time.time()
        self.ref_color, self.ref_color_cov, self.thr_mahalanobis = self.get_reference(
                orthomosaic=ortho,
                iterations=30,
                tile_size=2.5,
                segmentation_parameters=[120, 0.3, 3])
        proc_time = time.time() - start_time
        ic(self.ref_color)
        ic(self.ref_color_cov)
        ic(self.thr_mahalanobis)
        print('getting reference: ', proc_time)

        start_time = time.time()
        self.locate_pumpkins_in_orthomosaic(ortho)
        proc_time = time.time() - start_time
        print('segmentation, statistics and results generation: ', proc_time)


    def get_reference(self, orthomosaic, iterations, tile_size, segmentation_parameters):
        # segmentation_parameters: number of segments, compactness, no of std away for median to be considered a pumpkin)

        with rasterio.open(orthomosaic) as src:
            resolution = src.res
        resolution = np.round(resolution, 3)

        tile_size_height = int(tile_size / resolution[0])
        tile_size_width = int(tile_size / resolution[1])

        # define tiles
        overlap = 0
        tiles, _, _ = self.define_tiles(orthomosaic, overlap, tile_size_height, tile_size_width)
        tiles_to_check = self.get_non_empty_tiles(orthomosaic, tiles, iterations)

        segmented, inliers = self.establish_reference_color(orthomosaic,
                                                            tiles,
                                                            tiles_to_check,
                                                            segmentation_parameters)

        threshold_mahalanobis = self.establish_mahalanobis_threshold(orthomosaic, tiles, tiles_to_check,
                                                                self.reference_color, self.reference_color_covariance, inliers, segmented)

        return self.reference_color, self.reference_color_covariance, threshold_mahalanobis


    def define_tiles(self, image, overlap, height, width):

        with rasterio.open(image) as src:
            columns = src.width
            rows = src.height

        last_position = (rows - height, columns - width)

        n_height = np.ceil(rows / (height * (1 - overlap))).astype(int)
        n_width = np.ceil(columns / (width * (1 - overlap))).astype(int)

        step_height = np.trunc(last_position[0] / (n_height - 1)).astype(int)
        step_width = np.trunc(last_position[1] / (n_width - 1)).astype(int)

        tiles = []
        for r in range(0, n_height):
            for c in range(0, n_width - 1):
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


    def get_non_empty_tiles(self, orthomosaic, tiles, no_of_tiles):
        number = len(tiles)
        # find a set of random, non-empty tiles to establish reference
        selected_tiles = []
        while len(selected_tiles) < no_of_tiles:
            possibility = random.randint(0, number-1)
            if possibility not in selected_tiles:
                tile = tiles[possibility]
                tile_im = read_tile(orthomosaic, tile)
                if np.median(tile_im[:, :, 0]) != 255:
                    selected_tiles.append(possibility)

        return selected_tiles


    def establish_reference_color(self, orthomosaic, tiles, tiles_to_process, segmentation_parameters):
        esps = ExtractSaturatedPixelSegments()
        pixels_all = esps.main(orthomosaic, tiles, tiles_to_process, segmentation_parameters)

        self.reference_color = np.average(pixels_all[:, (0, 2)], axis=0)
        self.reference_color_covariance = np.cov(pixels_all[:, (0, 2)].transpose())

        return esps.segmented, esps.inliers


    def establish_mahalanobis_threshold(self, orthomosaic, tiles, tiles_to_process,
                                        ref_color, ref_color_covariance, inliers, segmented):
        pixels_in = []
        for number in tiles_to_process:
            tile = tiles[number]
            tile_im = read_tile(orthomosaic, tile)
            img_HLS = cv.cvtColor(tile_im, cv.COLOR_BGR2HLS)

            mahalanobis_distance = self.calculate_mahalanobis_distance(img_HLS[:, :, (0, 2)],
                                                                  ref_color, ref_color_covariance)
            seg_in = inliers[str(number)]
            tile_segments = segmented[str(number)]

            mask_in = np.ones_like(mahalanobis_distance)
            for seg in seg_in:
                mask_in[tile_segments == seg] = 0

            img_temp_in = mahalanobis_distance.copy().astype(float)
            img_temp_in[mask_in > 0] = -99

            temp_pixels_in = np.reshape(img_temp_in, -1).tolist()
            cleaned_temp_pixels_in = [x for x in temp_pixels_in if x >= 0]
            pixels_in.append(cleaned_temp_pixels_in)

        pixels_in = [item for sublist in pixels_in for item in sublist]
        return np.mean(pixels_in) + np.std(pixels_in)


    def calculate_mahalanobis_distance(self, image, reference_color, reference_color_covariance):
        pixels = np.reshape(image, (-1, 2))
        inv_cov = np.linalg.inv(reference_color_covariance)
        diff = pixels - reference_color
        moddotproduct = diff * (diff @ inv_cov)
        mahalanobis_dist = np.sum(moddotproduct, axis=1)
        mahalanobis_dist = np.sqrt(mahalanobis_dist)
        mahalanobis_distance_image_in_function = np.reshape(mahalanobis_dist, (image.shape[0], image.shape[1]))

        return mahalanobis_distance_image_in_function


    def locate_pumpkins_in_orthomosaic(self, ortho):
        with rasterio.open(ortho) as src:
            self.resolution = np.round(src.res, 3)
            self.crs = src.crs
            self.left = src.bounds[0]
            self.top = src.bounds[3]

        processing_tiles = self.get_processing_tiles(ortho, self.tile_size)

        for tile_number, tile in enumerate(processing_tiles):
            img_RGB = read_tile(ortho, tile)
            if self.is_image_empty(img_RGB):
                continue

            self.process_tile(ortho, img_RGB, tile_number, tile)

        pumpkins_with_attributes = self.extract_pumpkin_attributes(processing_tiles)

        mean_pumpkin = np.mean(pumpkins_with_attributes[:, 2])
        median_pumpkin = np.median(pumpkins_with_attributes[:, 2])
        std_pumpkin = np.std(pumpkins_with_attributes[:, 2])
        threshold_pumpkin_size = median_pumpkin + std_pumpkin
        for i in range(0, pumpkins_with_attributes.shape[0]):
            if pumpkins_with_attributes[i, 2] <= threshold_pumpkin_size:
                pumpkins_with_attributes[i, 4] = 1
            else:
                pumpkins_with_attributes[i, 4] = np.round(pumpkins_with_attributes[i, 2] / mean_pumpkin)

        file_name = ortho[:-4] + '_pumpkins_new' + '.txt'
        np.savetxt(file_name, pumpkins_with_attributes[:, (0, 1, 4)])
        count = int(np.sum(pumpkins_with_attributes[:, 4]))
        print(ortho, self.thr_mahalanobis, count)


    def get_processing_tiles(self, ortho, tile_size):
        processing_tiles, st_width, st_height = self.define_tiles(ortho, 0.01, tile_size, tile_size)

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
        return np.max(image[:, :, 0]) == np.min(image[:, :, 0])


    def process_tile(self, ortho, img_RGB, tile_number, tile):
        tile.ulc_global = [
                self.top - (tile.ulc[0] * self.resolution[0]), 
                self.left + (tile.ulc[1] * self.resolution[1])]

        pumpkins = []
        img = cv.cvtColor(img_RGB, cv.COLOR_BGR2HLS)

        mahalanobis_distance_image = self.calculate_mahalanobis_distance(img[:, :, (0, 2)], self.ref_color, self.ref_color_cov)

        _, segmented_image = cv.threshold(mahalanobis_distance_image, self.thr_mahalanobis, 255, cv.THRESH_BINARY)

        segmented_median_blurred = cv.medianBlur(segmented_image.astype(np.uint8), 5)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv.morphologyEx(segmented_median_blurred, cv.MORPH_ERODE, kernel)

        contours, _ = cv.findContours(image=dilated,
                                      mode=cv.RETR_TREE,
                                      method=cv.CHAIN_APPROX_NONE)

        contours_limited, pumpkins = self.keep_contours_within_processing_range(contours, tile)
        tile.pumpkin_list = pumpkins
        tile.contours = contours_limited

        if len(pumpkins) > 0:
            pumpkins_array = np.asarray(pumpkins)
            pumpkins_global = pumpkins_array.copy().astype(float)
            pumpkins_global[:, 0] = pumpkins_global[:, 0] * self.resolution[0] + tile.ulc_global[1]
            pumpkins_global[:, 1] = tile.ulc_global[0] - pumpkins_global[:, 1] * self.resolution[1]
            tile.pumpkins_global = pumpkins_global.tolist()

        # limit image to just boundaries and save it with georeference
        temp_ulc_global = [tile.ulc_global[0] - tile.processing_range[0][0] * self.resolution[0],
                           tile.ulc_global[1] + tile.processing_range[1][0]*self.resolution[0]]

        width = tile.processing_range[1][1] - tile.processing_range[1][0]
        height = tile.processing_range[0][1] - tile.processing_range[0][0]

        transform = Affine.translation(temp_ulc_global[1] + self.resolution[0] / 2, temp_ulc_global[0] - self.resolution[0] / 2) * \
                    Affine.scale(self.resolution[0], -self.resolution[0])

        # optional save of results - just lob detection and thresholding result
        self.save_results(img_RGB, pumpkins, tile, tile_number, dilated, ortho, self.resolution, height, width, self.crs, transform)


    def keep_contours_within_processing_range(self, contours, tile):
        pumpkins = []
        is_closed_contour = True
        contours_limited = []
        for contour in contours:
            area = cv.contourArea(contour, is_closed_contour)
            if area > 0:
                M = cv.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if cY >= tile.processing_range[0][0]:
                    if cY <= tile.processing_range[0][1]:
                        if cX >= tile.processing_range[1][0]:
                            if cX <= tile.processing_range[1][1]:
                                pumpkins.append([cX, cY])
                                contours_limited.append(contour)
        return contours_limited, pumpkins




    def save_results(self, img_RGB, pumpkins, tile, tile_number, dilated, ortho, res, height, width, crs, transform):
        img_to_annotate = img_RGB.copy()
        if len(pumpkins) > 0:
            for pumpkin in pumpkins:
                cX = pumpkin[0]
                cY = pumpkin[1]
                img_to_annotate = cv.circle(img_to_annotate, (cX, cY), 10, (0, 0, 255), 2)

        img_to_annotate = img_to_annotate[int(tile.processing_range[0][0]):int(tile.processing_range[0][1]),
                                          int(tile.processing_range[1][0]):int(tile.processing_range[1][1]), :]
        dilated = dilated[int(tile.processing_range[0][0]):int(tile.processing_range[0][1]),
                          int(tile.processing_range[1][0]):int(tile.processing_range[1][1])]

        name_annotated_image = ortho[:-4] + '/geo_tile_' + str(tile_number) + '.tiff'
        name_segmentation_results = ortho[:-4] + '/geo_tile_seg_' + str(tile_number) + '.tiff'

        img_to_save = cv.cvtColor(img_to_annotate, cv.COLOR_BGR2RGB)
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
        img_to_save = cv.merge((dilated, dilated, dilated))
        temp_to_save = img_to_save.transpose(2, 0, 1)
        new_dataset = rasterio.open(name_segmentation_results,
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


    def extract_pumpkin_attributes(self, processing_tiles):
        pumpkins = []
        contours = []
        for tile in processing_tiles:
            pumpkins.append(tile.pumpkins_global)
            contours.append(tile.contours)
        pumpkins = [item for sublist in pumpkins for item in sublist]
        contours = [item for sublist in contours for item in sublist]

        pumpkins = np.asarray(pumpkins)
        pumpkins_with_attributes = np.zeros((pumpkins.shape[0], 5))
        pumpkins_with_attributes[:, 0:2] = pumpkins

        for i, cnt in enumerate(contours):
            area = cv.contourArea(cnt)
            _, radius = cv.minEnclosingCircle(cnt)
            area_circle = 2 * np.pi * radius

            pumpkins_with_attributes[i, 2] = area
            pumpkins_with_attributes[i, 3] = (area/area_circle)

        return pumpkins_with_attributes





pc = PumpkinCounter()
pc.main(orthos)
