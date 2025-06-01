from typing import Optional, Dict, List, Tuple, Union, Callable
from string import Template
from contextlib import contextmanager
from importlib.resources import as_file, files, Package
import struct
import csv
import functools
import os
import sys
import datetime
import itertools
import logging
import re
from pathlib import Path
from threading import Semaphore
import math
import json
import warnings
logging.basicConfig(level=logging.INFO)

import fire
import sh
from sh import Command
import moody
import pyproj
import papermill as pm
import pvl

class HiRISE(object):
    r"""
    ASAP Stereo Pipeline - HiRISE workflow

    █████████████████████████████████████████████████████████████

              ___   _____ ___    ____
             /   | / ___//   |  / __ \
            / /| | \__ \/ /| | / /_/ /
           / ___ |___/ / ___ |/ ____/
          /_/  |_/____/_/  |_/_/      𝑆 𝑇 𝐸 𝑅 𝐸 𝑂

          pyameslib (0.3.1)

          Github: https://github.com/AndrewAnnex/asap_stereo
          Cite: https://doi.org/10.5281/zenodo.4171570

    █████████████████████████████████████████████████████████████
    """

    def __init__(self, https=False, datum="D_MARS", proj: Optional[str] = None):
        self.https = https
        self.cs = CommonSteps()
        self.datum = datum
        self.hiedr = sh.Command('hiedr2mosaic.py').bake(_log_msg=custom_log)
        self.cam2map = sh.Command('cam2map').bake(_log_msg=custom_log)
        self.cam2map4stereo = sh.Command('cam2map4stereo.py').bake(_log_msg=custom_log)
        # if proj is not none, get the corresponding proj or else override with proj,
        # otherwise it's a none so remain a none
        self.proj = self.cs.projections.get(proj, proj)
        # make the pipeline todo think about metasteps, can I do nested lists and flatten iter?
        self.pipeline = [
            self.step_1,
            self.step_2,
            self.step_3,
            self.step_4,
            self.step_5,
            self.step_6,
            self.step_6,
            self.step_7,
            self.step_8,
            self.step_9,
            self.step_10,
            self.step_11,
            self.step_12
        ]

    def get_hirise_emission_angle(self, pid: str) -> float:
        """
        Use moody to get the emission angle of the provided HiRISE image id

        :param pid: HiRISE image id
        :return: emission angle
        """

        return float(
            moody.ODE(self.https).get_hirise_meta_by_key(f'{pid}_R*', 'Emission_angle'))

    def get_hirise_order(self, one: str, two: str) -> Tuple[str, str]:
        """
        Get the image ids sorted by lower emission angle

        :param one: first image
        :param two: second image
        :return: tuple of sorted images
        """

        em_one = self.get_hirise_emission_angle(one)
        em_two = self.get_hirise_emission_angle(two)
        if em_one <= em_two:
            return one, two
        else:
            return two, one

    def generate_hirise_pair_list(self, one, two):
        """
        Generate the hirise pair.lis file for future steps

        :param one: first image id
        :param two: second image id
        """
        order = self.get_hirise_order(one, two)
        with open('pair.lis', 'w', encoding='utf') as o:
            for pid in order:
                o.write(pid)
                o.write('\n')

    @staticmethod
    def notebook_pipeline_make_dem(left: str,
                                   right: str,
                                   config: str,
                                   ref_dem: str,
                                   gcps: str = '',
                                   max_disp: float = None,
                                   downsample: int = None,
                                   dem_gsd: float = 1.0,
                                   img_gsd: float = 0.25,
                                   max_ba_iterations: int = 200,
                                   alignment_method='rigid', step_kwargs=None,
                                   working_dir='./', out_notebook=None, **kwargs):
        """
        First step in HiRISE DEM pipeline that uses papermill to persist log

        This command does most of the work, so it is long running!
        I recommend strongly to use nohup with this command, even more so for HiRISE!

        :param out_notebook: output notebook log file name, defaults to log_asap_notebook_pipeline_make_dem_hirise.ipynb
        :param working_dir: Where to execute the processing, defaults to current directory
        :param config:  ASP config file to use for processing
        :param left: first image id
        :param right: second image id
        :param alignment_method: alignment method to use for pc_align
        :param downsample: Factor to downsample images for faster production
        :param ref_dem: path to reference DEM or PEDR csv file
        :param gcps: path to gcp file todo: currently only one gcp file allowed
        :param max_disp: Maximum expected displacement in meters, specify none to determine it automatically
        :param dem_gsd: desired GSD of output DEMs (4x image GSD)
        :param img_gsd: desired GSD of output ortho images
        :param max_ba_iterations: maximum number of BA steps to use per run (defaults to 50 for slow running hirise BA)
        :param step_kwargs: Arbitrary dict of kwargs for steps following {'step_#' : {'key': 'value}}
        """
        if not out_notebook:
            out_notebook = f'{working_dir}/log_asap_notebook_pipeline_make_dem_hirise.ipynb'
        if 'postfix' in kwargs.keys():
            resource = 'asap_hirise_workflow_hiproc.ipynb'
        else:
            resource = 'asap_hirise_workflow_nomap.ipynb'
        with as_file(files('pyameslib').joinpath(resource)) as src:
            pm.execute_notebook(
                src,
                out_notebook,
                parameters={
                    'left': left,
                    'right': right,
                    'config': config,
                    'output_path': working_dir,
                    'ref_dem': ref_dem,
                    'gcps': gcps,
                    'max_disp': max_disp,
                    'dem_gsd': dem_gsd,
                    'img_gsd': img_gsd,
                    'alignment_method': alignment_method,
                    'downsample': downsample,
                    'max_ba_iterations': max_ba_iterations,
                    'postfix': kwargs.pop('postfix', '_RED.cub'),
                    'step_kwargs': step_kwargs
                },
                request_save_on_cell_execute=True,
                **kwargs
            )

    @rich_logger
    def step_1(self, one, two, cwd: Optional[str] = None):
        """
        Download HiRISE EDRs

        Download two HiRISE images worth of EDR files to two folders

        :param one: first image id
        :param two: second image id
        :param cwd:
        """
        with cd(cwd):
            self.generate_hirise_pair_list(one, two)
            # download files
            Path(one).mkdir(exist_ok=True)
            with cd(one):
                moody.ODE(self.https).download_hirise_edr(f'{one}_R*')
            Path(two).mkdir(exist_ok=True)
            with cd(two):
                moody.ODE(self.https).download_hirise_edr(f'{two}_R*')

    @rich_logger
    def step_2(self):
        """
        Metadata init

        Create various files with info for later steps

        """
        self.cs.create_stereopairs_lis()
        self.cs.create_stereodirs_lis()
        self.cs.create_stereodirs()
        self.cs.create_stereopair_lis()

    @rich_logger
    def step_3(self):
        """
        Hiedr2mosaic preprocessing

        Run hiedr2mosaic on all the data

        """

        def hiedr2mosaic(*im):
            # hiedr2moasic is given a glob of tifs
            pool.acquire()
            return self.hiedr(*im, '--threads', _threads_singleprocess, _bg=True, _done=done)

        left, right, both = self.cs.parse_stereopairs()
        procs = []
        with cd(Path(left)):
            procs.append(hiedr2mosaic(*list(Path('pyameslib/').glob('*.IMG'))))
        with cd(Path(right)):
            procs.append(hiedr2mosaic(*list(Path('pyameslib/').glob('*.IMG'))))
        _ = [p.wait() for p in procs]

    @rich_logger
    def step_4(self, postfix='*.mos_hijitreged.norm.cub', camera_postfix='_RED.json'):
        """
        Copy hieder2mosaic files

        Copy the hiedr2mosaic output to the location needed for cam2map4stereo

        :param postfix: postfix for cub files to use
        """
        left, right, both = self.cs.parse_stereopairs()
        both = Path(both)
        left_file = next(Path(f'./{left}/').glob(f'{left}{postfix}')).absolute()
        right_file = next(Path(f'./{right}/').glob(f'{right}{postfix}')).absolute()
        sh.ln('-s', left_file, (both / f'{left}_RED.cub').absolute())
        sh.ln('-s', right_file, (both / f'{right}_RED.cub').absolute())
        self.cs.generate_csm(postfix='_RED.cub', camera_postfix=camera_postfix)

    @rich_logger
    def step_5(self, refdem=None, gsd: float = None, postfix='_RED.cub',
               camera_postfix='_RED.json', bundle_adjust_prefix=None, **kwargs):
        """
        # todo this no longer makes sense for step 5, needs to run after bundle adjust but before stereo
        # todo need cameras by this point, currently done in BA
        Map project HiRISE data for stereo processing

        Note this step is optional.

        :param bundle_adjust_prefix:
        :param camera_postfix:
        :param postfix: postfix for cub files to use
        :param gsd: override for final resolution in meters per pixel (mpp)
        """
        return self.cs.mapproject_both(refdem=refdem, mpp=gsd, postfix=postfix,
                                       camera_postfix=camera_postfix,
                                       bundle_adjust_prefix=bundle_adjust_prefix, **kwargs)

    @rich_logger
    def step_6(self, *vargs, postfix='_RED.cub', camera_postfix='_RED.json',
               bundle_adjust_prefix='adjust/ba', **kwargs) -> sh.RunningCommand:
        """
        Bundle Adjust HiRISE

        Run bundle adjustment on the HiRISE map projected data

        :param postfix: postfix for cub files to use
        :param camera_postfix: postfix for cameras to use
        :param vargs: variable length additional positional arguments to pass to bundle adjust
        :param bundle_adjust_prefix:
        """
        return self.cs.bundle_adjust(*vargs, postfix=postfix, camera_postfix=camera_postfix,
                                     bundle_adjust_prefix=bundle_adjust_prefix, **kwargs)

    @rich_logger
    def step_7(self, stereo_conf, postfix='_RED.cub', camera_postfix='_RED.json',
               run='results_ba', posargs='', **kwargs):
        """
        Parallel Stereo Part 1

        Run first part of parallel_stereo

        :param run: folder for results of run
        :param postfix: postfix for cub files to use
        :param camera_postfix: postfix for cameras to use
        """
        return self.cs.stereo_asap(stereo_conf, run=run, cub_postfix=postfix,
                                   cam_postfix=camera_postfix, posargs=posargs,
                                   **{**self.cs.defaults_ps1, **kwargs})

    @rich_logger
    def step_8(self, stereo_conf, postfix='_RED.cub', camera_postfix='_RED.json',
               run='results_ba', posargs='', **kwargs):
        """
        Parallel Stereo Part 2

        Run second part of parallel_stereo, stereo is completed after this step

        :param run: folder for results of run
        :param postfix: postfix for cub files to use
        :param camera_postfix: postfix for cameras to use
        """
        return self.cs.stereo_asap(stereo_conf, run=run, cub_postfix=postfix,
                                   cam_postfix=camera_postfix, posargs=posargs,
                                   **{**self.cs.defaults_ps2, **kwargs})

    @rich_logger
    def step_9(self, mpp=2, just_dem=True, postfix='_RED.cub', run='results_ba', **kwargs):
        """
        Produce preview DEMs/Orthos

        Produce dem from point cloud, by default 2mpp for hirise for max-disparity estimation

        :param run: folder for results of run
        :param postfix: postfix for cub files to use
        :param just_dem: set to True if you only want the DEM and no other products like the ortho and error images
        :param mpp:
        """
        just_ortho = not just_dem
        return self.cs.point_to_dem(mpp,
                                    'PC.tif',
                                    just_ortho=just_ortho,
                                    cub_postfix=postfix,
                                    run=run,
                                    kind='align',
                                    use_proj=self.proj,
                                    **kwargs)

    def _gdal_hirise_rescale(self, mpp, postfix='_RED.cub', run='results_ba', **kwargs):
        """
        resize hirise image using gdal_translate

        :param postfix: postfix for cub files to use
        :param mpp:
        """
        left, right, both = self.cs.parse_stereopairs()
        mpp_postfix = self.cs.get_mpp_postfix(mpp)
        with cd(Path.cwd() / both / run / 'dem'):
            # check the GSD against the MPP
            self.cs.check_mpp_against_true_gsd(f'../../{left}{postfix}', mpp)
            in_dem = next(Path.cwd().glob(
                '*-DEM.tif'))  # todo: this might not always be the right thing to do...
            return sh.gdal_translate('-r', 'cubic', '-tr', float(mpp), float(mpp), in_dem,
                                     f'./{both}_{mpp_postfix}-DEM.tif')

    @rich_logger
    def pre_step_10(self, refdem, run='results_ba', alignment_method='translation',
                    do_resample='gdal', **kwargs):
        """
        Hillshade Align before PC Align

        Automates the procedure to use ipmatch on hillshades of downsampled HiRISE DEM
        to find an initial transform

        :param run:
        :param do_resample:  can be: 'gdal' or 'asp' or anything else for no resampling
        :param alignment_method: can be 'similarity' 'rigid' or 'translation'
        :param refdem: path to reference DEM or PEDR csv file
        :param kwargs:
        """
        left, right, both = self.cs.parse_stereopairs()
        defaults = {
            '--max-displacement': -1,
            '--num-iterations': '0',
            '--ipmatch-options': '--debug-image',
            '--ipfind-options': '--ip-per-image 3000000 --ip-per-tile 8000 --interest-operator sift --descriptor-generator sift --debug-image 2',
            '--threads': _threads_singleprocess,
            '--initial-transform-from-hillshading': alignment_method,
            '--datum': self.datum,
            '--output-prefix': 'hillshade_align/out'
        }
        refdem = Path(refdem).absolute()
        refdem_mpp = math.ceil(self.cs.get_image_gsd(refdem))
        refdem_mpp_postfix = self.cs.get_mpp_postfix(refdem_mpp)
        # create the lower resolution hirise dem to match the refdem gsd
        if do_resample.lower() == 'asp':
            # use the image in a call to pc_align with hillshades, slow!
            self.step_9(mpp=refdem_mpp, just_dem=True, **kwargs)
        elif do_resample.lower() == 'gdal':
            # use gdal translate to resample hirise dem down to needed resolution first for speed
            self._gdal_hirise_rescale(refdem_mpp, **kwargs)
        else:
            print('Not resampling HiRISE per user request')
        # TODO: auto crop the reference dem to be around hirise more closely
        with cd(Path.cwd() / both / run):
            kwargs.pop('postfix', None)
            lr_hirise_dem = Path.cwd() / 'dem' / f'{both}_{refdem_mpp_postfix}-DEM.tif'
            args = kwargs_to_args({**defaults, **clean_kwargs(kwargs)})
            cmd_res = self.cs.pc_align(*args, lr_hirise_dem, refdem)
            # done! log out to user that can use the transform
        return '--initial-transform hillshade_align/out-transform.txt'

    @rich_logger
    def pre_step_10_pedr(self, pedr_list=None, postfix='_RED.cub') -> str:
        """
        Use MOLA PEDR data to align the HiRISE DEM to in case no CTX DEM is available

        :param pedr_list: path local PEDR file list, default None to use REST API
        :param postfix: postfix for cub files to use
        """
        return self.cs.get_pedr_4_pcalign_common(postfix, self.proj, self.https,
                                                 pedr_list=pedr_list)

    @rich_logger
    def step_10(self, maxd, refdem, run='results_ba', highest_accuracy=True, **kwargs):
        """
        PC Align HiRISE

        Run pc_align using provided max disparity and reference dem
        optionally accept an initial transform via kwargs

        :param run:
        :param maxd: Maximum expected displacement in meters
        :param refdem: path to reference DEM or PEDR csv file
        :param highest_accuracy: use highest precision alignment (more memory and cpu intensive)
        :param kwargs: kwargs to pass to pc_align, use to override ASAP defaults
        """
        # run any pre-step 10 steps needed
        if 'with_pedr' in kwargs:
            refdem = self.pre_step_10_pedr(pedr_list=kwargs.get('pedr_list', None),
                                           postfix=kwargs.get('postfix', '_RED.cub'))
        elif 'with_hillshade_align' in kwargs:
            cmd = self.pre_step_10(refdem,
                                   **kwargs)  # todo check that this blocks until finished
            kwargs['--initial_transform'] = 'hillshade_align/out-transform.txt'
        else:
            pass

        return self.cs.point_cloud_align(self.datum, maxd=maxd, refdem=refdem,
                                         highest_accuracy=highest_accuracy, run=run,
                                         kind='align', **kwargs)

    @rich_logger
    def step_11(self, mpp=1.0, just_ortho=False, postfix='_RED.cub', run='results_ba',
                output_folder='dem_align', **kwargs):
        """
        Produce final DEMs/Orthos

        Run point2dem on the aligned output to produce final science ready products

        :param run:
        :param postfix: postfix for cub files to use
        :param mpp: Desired GSD (meters per pixel)
        :param just_ortho: if True, just render out the ortho images
        :param output_folder: output folder name
        :param kwargs: any other kwargs you want to pass to point2dem
        """
        return self.cs.point_to_dem(mpp,
                                    'trans_reference.tif',
                                    just_ortho=just_ortho,
                                    cub_postfix=postfix,
                                    run=run,
                                    kind='align',
                                    use_proj=self.proj,
                                    output_folder=output_folder,
                                    **kwargs)

    @rich_logger
    def step_12(self, run='results_ba', output_folder='dem_align', **kwargs):
        """
        Adjust DEM to geoid

        Run geoid adjustment on dem for final science ready product

        :param run:
        :param output_folder:
        :param kwargs:
        """
        return self.cs.geoid_adjust(run=run, output_folder=output_folder, **kwargs)