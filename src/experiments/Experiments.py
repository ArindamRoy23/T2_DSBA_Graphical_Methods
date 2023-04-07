"""

TODO:

    Clean this file

"""

from ..lib.utils.FileHandling.FileHandlingInterface import *
from ..lib.SVCDSeg.SVCD import *
from tqdm import tqdm

import matplotlib.pyplot as plt
import logging
import time
import warnings
import re
import os


class Experiment(object):

    @staticmethod
    def run_segmentation(
            SAMPLE_IMAGE_PATH: str, 
            SAMPLE_SCRIBBLE_PATH: str,
            max_iter: int = 1000,
            lambda_: float = 8e-4,
            gamma: float = 5e-0, 
            alpha: float = 18e-1, # width of spatial kernel (as in the paper)
            sigma: float = 13e-1, # width of chromatic kernel (as in the paper)
            tau_primal: float = 25e-2, 
            tau_dual: float = 5e-1,
            early_stop: bool = False,
            tolerance: float = 1e-5,
            use_tqdm: bool = True,  
            debug: int = 0,
            return_: int = 0,
            transpose: bool = False 
        ) -> SVCDSeg:
        """
        
        """
        sample_image_kea = TargetImage(
            SAMPLE_IMAGE_PATH
        )
        sample_scribble_kea = EncodedScribble(
            SAMPLE_SCRIBBLE_PATH,
        )
        n_classes = sample_scribble_kea.get_n_classes()
        segmenter = SVCDSeg(
            n_classes,
            max_iter = max_iter,
            lambda_ = 8e-3,
            gamma = gamma, 
            alpha = alpha, 
            sigma = sigma, 
            tau_primal = tau_primal, 
            tau_dual = tau_dual, 
            early_stop = early_stop, 
            tolerance = tolerance, 
            use_tqdm = use_tqdm, 
            debug = debug, 
            return_ = return_, 
            transpose = transpose
        )
        segmenter.fit(
            sample_image_kea,
            sample_scribble_kea
        )
        # print('Segmenter run ok')
        return segmenter

    @staticmethod
    def find_image_id(
            SAMPLE_IMAGE_PATH: str
        ) -> str:
        """
        
        """
        image_id = re.findall(".*(\d*_\d*)\.\w*", SAMPLE_IMAGE_PATH)[0]
        return image_id

    
    
    @staticmethod
    def save_segmentation_mask(
            segmentation_mask: np.ndarray,
            image_id: str,  
            DUMP_DIR: str = r"./plots/",
        ) -> None:
        """
        
        """
        try:
            os.mkdir(path_dir)

        except:
            print("directory already existing, skipping its creation")
        path_dir = f"{DUMP_DIR}/{image_id}_"
        fig, ax = plt.subplots()
        im1 = ax.imshow(segmentation_mask, cmap='magma', alpha=1.0)
        ax.set_title("Segmentation Result")
        seg_result_file = f"{path_dir}segmentation_result.png"
        fig.savefig(seg_result_file)
        # print(f'seg_result_file: {seg_result_file}')
    
    @staticmethod
    def save_segmentation(
            SAMPLE_IMAGE_PATH: str, 
            SAMPLE_SCRIBBLE_PATH: str,
            DUMP_DIR: str = r"./plots/",
            max_iter: int = 1000,
            lambda_: float = 8e-4
        ) -> None:
        """
        
        """
        sample_image_kea = TargetImage(
            SAMPLE_IMAGE_PATH
        )
        sample_scribble_kea = EncodedScribble(
            SAMPLE_SCRIBBLE_PATH,
        )
        image_id = re.findall(".*(\d*_\d*)\.\w*", SAMPLE_IMAGE_PATH)[0]
        path_dir = f"{DUMP_DIR}{image_id}_lambda_{lambda_}/"
        print(path_dir)
        os.mkdir(path_dir)
        segmenter = run_segmentation(
            SAMPLE_IMAGE_PATH, 
            SAMPLE_SCRIBBLE_PATH,
            lambda_ = lambda_,
            max_iter = max_iter
        )
        final_segmentation = np.argmax(segmenter.theta_t1, axis = 0).transpose(1, 0)
        fig, ax = plt.subplots()
        im1 = ax.imshow(final_segmentation, cmap='magma', alpha=1.0)
        ax.set_title("Segmentation Result")
        seg_result_file = f"{path_dir}segmentation_result.png"
        print(f"seg_result_file {seg_result_file}")
        fig.savefig(seg_result_file)
        for class_ in range(segmenter.xi.shape[1]):
            xi = segmenter.xi[:, class_, :, :]
            fig, ax = plt.subplots()
            dual_vars = np.sum(xi**2, axis = 0)
            dual_vars = np.sqrt(dual_vars).transpose(1, 0)
            vmin = np.min(dual_vars)
            vmax = np.max(dual_vars)
            im1 = ax.imshow(dual_vars, cmap='magma', alpha=1.0, vmin=vmin, vmax=vmax)
            ax.set_title("Dual Variables")
            fig.colorbar(im1, ax=ax, label='')
            dual_result_file = f"{path_dir}dual_variable_{class_}.png"
            fig.savefig(dual_result_file)
        fig, ax = plt.subplots()

        im_dual = ax.plot(segmenter.dual_energy_history, label = "Dual Energy")
        im_primal = ax.plot(segmenter.primal_energy_history, label = "Primal Energy")
        ax.set_title("Primal-Dual energy history")
        ax.legend()
        energy_history_result_file = f"{path_dir}energy_history.png"
        fig.savefig(energy_history_result_file)

        im_arr = sample_image_kea.get_image_array()

        im_shape = sample_image_kea.get_image_shape()
        segmenter.utils.make_derivative_matrix(*im_shape[1::-1])
        half_g = segmenter.halfg
        _, w, h = half_g.shape
        half_g = half_g.reshape(w, h)
        half_g.shape
        vmin = np.min(half_g)
        vmax = np.max(half_g)

        fig, ax = plt.subplots()

        im1 = ax.imshow(half_g.transpose(1, 0), cmap='magma', alpha=1.0, vmin=vmin, vmax=vmax)
        ax.set_title("G function on the image")
        fig.colorbar(im1, ax=ax, label='')
        g_function_file = f"{path_dir}g_function_{class_}.png"
        fig.savefig(g_function_file)
        for idx, likelihood in enumerate(segmenter.fitted_likelihood):
            fig, ax = plt.subplots()
            vmin = np.min(likelihood)
            vmax = np.max(likelihood)
            im1 = ax.imshow(likelihood.transpose(1, 0), cmap='magma', alpha=1.0, vmin=vmin, vmax=vmax)
            fig.colorbar(im1, ax=ax, label='')
            g_function_file = f"{path_dir}likelihood_{class_}.png"
            fig.savefig(g_function_file)

    @staticmethod
    def save_segmentations_folder(
            data_folder: str = "./data/demoPascal/",
            lambdas: list = ["8e-3", "8e-4", "8e-5"],
            max_iter = 1000
        ) -> None:
        list_dir = os.listdir(data_folder)
        print("list dir")
        for idx in tqdm(range(12, len(list_dir) - 1)):
            image_path = list_dir[idx]
            image_path = os.path.join(data_folder, image_path)
            scribble_path = list_dir[idx + 1]
            scribble_path = os.path.join(data_folder, scribble_path)
            print(f"image_path {image_path}\n scribble_path {scribble_path}")
            for lambda_ in lambdas:
                save_segmentation(
                    image_path, 
                    scribble_path, 
                    lambda_ = float(lambda_),
                    max_iter = max_iter
                )