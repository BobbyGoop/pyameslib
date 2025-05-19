# BSD 3-Clause License
#
# Copyright (c) 2020-2021, Andrew Michael Annex
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
import src.moody
import pyproj
import papermill as pm
import pvl



class ASAP(object):
    r"""
    ASAP Stereo Pipeline

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

    def __init__(self, https=False, datum="D_MARS"):
        self.https  = https
        self.hirise = HiRISE(self.https, datum=datum)
        self.ctx    = CTX(self.https, datum=datum)
        self.common = CommonSteps()
        self.georef = Georef()
        self.get_srs_info = self.common.get_srs_info
        self.get_map_info = self.common.get_map_info

    def ctx_one(self, left, right, cwd: Optional[str] = None):
        """
        Run first stage of CTX pipeline

        This command runs steps 1-3 of the CTX pipeline

        :param left: left image id
        :param right: right image id
        :param cwd: directory to run process within (default to CWD)
        """
        with cd(cwd):
            self.ctx.step_1(left, right)
            # ctxedr2lev1eo steps
            self.ctx.step_2()
            # move things
            self.ctx.step_3()

    def ctx_two(self, stereo: str, pedr_list: str, stereo2: Optional[str] = None, cwd: Optional[str] = None) -> None:
        """
        Run Second stage of CTX pipeline

        This command runs steps 4-12 of the CTX pipeline

        :param stereo: ASP stereo config file to use
        :param pedr_list: Path to PEDR files, defaults to None to use ODE Rest API
        :param stereo2: 2nd ASP stereo config file to use, if none use first stereo file again
        :param cwd: directory to run process within (default to CWD)
        """
        with cd(cwd):
            self.ctx.step_4()
            self.ctx.step_5(stereo)
            self.ctx.step_6(stereo)
            self.ctx.step_7(mpp=100, just_ortho=False, dem_hole_fill_len=50)
            self.ctx.step_8()
            self.ctx.step_9()
            self.ctx.step_10(stereo2 if stereo2 else stereo)
            self.ctx.step_11(stereo2 if stereo2 else stereo)
            self.ctx.step_7(run='results_map_ba')
            self.ctx.step_8(run='results_map_ba')
            self.ctx.step_12(pedr_list)

    def ctx_three(self, max_disp: float = None, demgsd: float = 24, imggsd: float = 6, cwd: Optional[str] = None, **kwargs) -> None:
        """
        Run third and final stage of the CTX pipeline

        This command runs steps 13-15 of the CTX pipeline

        :param max_disp: Maximum expected displacement in meters
        :param demgsd: GSD of final Dem, default is 1 mpp
        :param imggsd: GSD of full res image
        :param cwd: directory to run process within (default to CWD)
        :param kwargs:
        """
        with cd(cwd):
            self.ctx.step_13(max_disp, **kwargs)
            self.ctx.step_14(mpp=demgsd, **kwargs)
            self.ctx.step_15(**kwargs)
            # go back and make final orthos and such
            self.ctx.step_14(mpp=imggsd, just_ortho=True, **kwargs)
            self.ctx.step_8(run='results_map_ba', output_folder='dem_align')

    def hirise_one(self, left, right):
        """
        Download the EDR data from the PDS, requires two HiRISE Id's
        (order left vs right does not matter)

        This command runs step 1 of the HiRISE pipeline

        :param left: HiRISE Id
        :param right: HiRISE Id
        """
        self.hirise.step_1(left, right)

    def hirise_two(self, stereo, mpp=2, bundle_adjust_prefix='adjust/ba', max_iterations=50) -> None:
        """
        Run various calibration steps then:
        bundle adjust, produce DEM, render low res version for inspection
        This will take a while (sometimes over a day), use nohup!

        This command runs steps 2-9 of the HiRISE pipeline

        :param stereo: ASP stereo config file to use
        :param mpp: preview DEM GSD, defaults to 2 mpp
        :param bundle_adjust_prefix: bundle adjust prefix, defaults to 'adjust/ba'
        :param max_iterations: number of iterations for HiRISE bundle adjustment, defaults to 50
        """
        self.hirise.step_2()
        self.hirise.step_3()
        self.hirise.step_4()
        self.hirise.step_5()
        self.hirise.step_6(bundle_adjust_prefix=bundle_adjust_prefix, max_iterations=max_iterations)
        self.hirise.step_7(stereo)
        self.hirise.step_8(stereo)
        self.hirise.step_9(mpp=mpp)

    def hirise_three(self, max_disp, ref_dem, demgsd: float = 1, imggsd: float = 0.25, **kwargs) -> None:
        """
        Given estimate of max disparity between reference elevation model
        and HiRISE output, run point cloud alignment and
        produce the final DEM/ORTHO data products.

        This command runs steps 10-12 of the HiRISE pipeline

        :param max_disp: Maximum expected displacement in meters
        :param ref_dem: Absolute path the reference dem
        :param demgsd: GSD of final Dem, default is 1 mpp
        :param imggsd: GSD of full res image
        """
        self.hirise.step_10(max_disp, ref_dem, **kwargs)
        self.hirise.step_11(mpp=demgsd, **kwargs)
        self.hirise.step_12(**kwargs)
        # if user wants a second image with same res as step
        # eleven don't bother as prior call to eleven did the work
        if not math.isclose(imggsd, demgsd):
            self.hirise.step_11(mpp=imggsd, just_ortho=True)

    def info(self):
        """
        Get the number of threads and processes as a formatted string

        :return: str rep of info
        """
        return f"threads sp: {_threads_singleprocess}\nthreads mp: {_threads_multiprocess}\nprocesses: {_processes}"


# def main():
#     fire.Fire(ASAP)
#
# if __name__ == '__main__':
#     main()

