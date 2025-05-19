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

from src.asp_wrapper import CommonSteps
from src.utils.common import _threads_singleprocess, get_affine_from_file

logging.basicConfig(level=logging.INFO)

import fire
import sh
from sh import Command
import moody
import pyproj
import papermill as pm
import pvl


class Georef(object):
    r"""
    ASAP Stereo Pipeline - Georef Tools

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

    @staticmethod
    def _read_ip_record(mf):
        """
        refactor of https://github.com/friedrichknuth/bare/ utils to use struct
        source is MIT licenced https://github.com/friedrichknuth/bare/blob/master/LICENSE.rst
        :param mf:
        :return:
        """
        x, y = struct.unpack("ff", mf.read(8))
        xi, yi = struct.unpack("ff", mf.read(8))
        orientation, scale, interest = struct.unpack("fff", mf.read(12))
        (polarity,) = struct.unpack("?", mf.read(1))
        octave, scale_lvl = struct.unpack("II", mf.read(8))
        ndesc = struct.unpack("Q", mf.read(8))[0]
        desc_len = int(ndesc * 4)
        desc_fmt = "f" * ndesc
        desc = struct.unpack(desc_fmt, mf.read(desc_len))
        iprec = [
            x,
            y,
            xi,
            yi,
            orientation,
            scale,
            interest,
            polarity,
            octave,
            scale_lvl,
            ndesc,
            *desc,
        ]
        return iprec

    @staticmethod
    def _read_match_file(filename):
        """
        refactor of https://github.com/friedrichknuth/bare/ utils to use struct
        source is MIT licenced https://github.com/friedrichknuth/bare/blob/master/LICENSE.rst
        :param filename:
        :return:
        """
        with open(filename, "rb") as mf:
            size1 = struct.unpack("q", mf.read(8))[0]
            size2 = struct.unpack("q", mf.read(8))[0]
            im1_ip = [Georef._read_ip_record(mf) for _ in range(size1)]
            im2_ip = [Georef._read_ip_record(mf) for _ in range(size2)]
            for i in range(len(im1_ip)):
                #'col1 row1 col2 row2'
                # todo: either here or below I may be making a mistaken row/col/x/y swap
                yield (im1_ip[i][0], im1_ip[i][1], im2_ip[i][0], im2_ip[i][1])

    @staticmethod
    def _read_match_file_csv(filename):
        # returns col, row
        with open(filename, "r") as src:
            return [list(map(float, _)) for _ in list(csv.reader(src))[1:]]

    @staticmethod
    def _read_gcp_file_csv(filename):
        with open(filename, "r") as src:
            return list(csv.reader(src))[1:]

    def __init__(self):
        self.cs = CommonSteps()

    def match_gsds(self, ref_image, *images):
        ref_gsd = int(self.cs.get_image_gsd(ref_image))
        for img in images:
            out_name = Path(img).stem + f"_{ref_gsd}.vrt"
            _ = sh.gdal_translate(
                img, out_name, "-of", "vrt", "-tr", ref_gsd, ref_gsd, "-r", "cubic"
            )
            yield out_name

    def normalize(self, image):
        # iterable of bands
        band_stats = self.cs.get_image_band_stats(image)
        # make output name
        out_name = Path(image).stem + "_normalized.vrt"
        # get bands scaling iterable, multiply by 1.001 for a little lower range
        scales = itertools.chain(
            (
                (
                    f'-scale_{bandif["band"]}',
                    float(bandif["minimum"]) * 1.001,
                    float(bandif["maximum"]) * 1.001,
                    1,
                    255,
                )
                for bandif in band_stats
            )
        )
        scales = [str(_).strip("'()\"") for _ in scales]
        # run gdal translate
        _ = sh.gdal_translate(
            image,
            out_name,
            "-of",
            "vrt",
            "-ot",
            "Byte",
            *scales,
            "-a_nodata",
            0,
            _out=sys.stdout,
            _err=sys.stderr,
        )
        return out_name

    def find_matches(
        self, reference_image, *mobile_images, ipfindkwargs=None, ipmatchkwargs=None
    ):
        """
        Generate GCPs for a mobile image relative to a reference image and echo to std out
        #todo: do we always assume the mobile_dem has the same srs/crs and spatial resolution as the mobile image?
        #todo: implement my own normalization
        :param reference_image: reference vis image
        :param mobile_images: image we want to move to align to reference image
        :param ipfindkwargs: override kwargs for ASP ipfind
        :param ipmatchkwargs: override kwarge for ASP ipmatch
        :return:
        """
        if ipfindkwargs is None:
            # todo --output-folder
            ipfindkwargs = f"--num-threads {_threads_singleprocess} --normalize --debug-image 1 --ip-per-tile 50"
        ipfindkwargs = ipfindkwargs.split(" ")
        # set default ipmatchkwargs if needed
        if ipmatchkwargs is None:
            ipmatchkwargs = "--debug-image --ransac-constraint homography"
        ipmatchkwargs = ipmatchkwargs.split(" ")
        # run ipfind on the reference image
        self.cs.ipfind(*ipfindkwargs, reference_image)
        # get vwip file
        ref_img_vwip = Path(reference_image).with_suffix(".vwip").absolute()
        for mobile_image in mobile_images:
            # run ipfind on the mobile image
            self.cs.ipfind(*ipfindkwargs, mobile_image)
            # get vwip file
            mob_img_vwip = Path(mobile_image).with_suffix(".vwip").absolute()
            # run ipmatch
            output_prefix = f"{Path(reference_image).stem}__{Path(mobile_image).stem}"
            self.cs.ipmatch(
                *ipmatchkwargs,
                reference_image,
                ref_img_vwip,
                mobile_image,
                mob_img_vwip,
                "--output-prefix",
                f"./{output_prefix}",
            )
            # done, todo return tuple of vwip/match files
            yield f"{output_prefix}.match"

    def matches_to_csv(self, match_file):
        """
        Convert an ASP .match file from ipmatch to CSV
        """
        matches = self._read_match_file(match_file)
        filename_out = os.path.splitext(match_file)[0] + ".csv"
        with open(filename_out, "w") as out:
            writer = csv.writer(out, delimiter=",")
            writer.writerow(["col1", "row1", "col2", "row2"])
            writer.writerows(matches)
        return filename_out

    def transform_matches(self, match_file_csv, mobile_img, mobile_other, outname=None):
        """
        Given a csv match file of two images (reference and mobile), and a third image (likely a DEM)
        create a modified match csv file with the coordinates transformed for the 2nd (mobile) image
        This works using the CRS of the images and assumes that both mobile images are already co-registered
        This is particularly useful when the imagery is higher pixel resolution than a DEM, and
        permits generating duplicated gcps
        """
        mp_for_mobile_img = self._read_match_file_csv(match_file_csv)
        img_t = get_affine_from_file(mobile_img)
        oth_t = get_affine_from_file(mobile_other)
        # todo: either here or below I may be making a mistaken row/col/x/y swap
        mp_for_other = [
            [*_[0:2], *(~oth_t * (img_t * _[2:4]))] for _ in mp_for_mobile_img
        ]
        # write output csv
        if not outname:
            outname = Path(match_file_csv).name.replace(
                Path(mobile_img).stem, Path(mobile_other).stem
            )
        with open(outname, "w") as out:
            writer = csv.writer(out, delimiter=",")
            writer.writerow(["col1", "row1", "col2", "row2"])
            writer.writerows(mp_for_other)
        return outname

    def create_gcps(self, reference_image, match_file_csv, out_name=None):
        """
        Given a reference image and a match file in csv format,
        generate a csv of GCPs. By default just prints to stdout
        but out_name allows you to name the csv file or you can pipe
        """
        # get reference affine transform to get world coords (crs coords) of reference rows/cols
        ref_t = get_affine_from_file(reference_image)
        # iterable of [ref_col, ref_row, mob_col, mob_row]
        mp_for_mobile_img = self._read_match_file_csv(match_file_csv)
        # get reference image matchpoint positions in reference crs
        # todo: either here or below I may be making a mistaken row/col/x/y swap
        mp_in_ref_crs = [ref_t * (c, r) for c, r, _, _ in mp_for_mobile_img]
        # get gcps which are tuples of [x1, y1, crs_x, crs_y]
        # I lob off the first two ip_pix points which are the reference row/col, I want mobile row/col
        gcps = [
            [*ip_pix[2:], *ip_crs]
            for ip_pix, ip_crs in zip(mp_for_mobile_img, mp_in_ref_crs)
        ]
        # get output file or stdout
        out = open(out_name, "w") if out_name is not None else sys.stdout
        w = csv.writer(out, delimiter=",")
        w.writerow(["col", "row", "easting", "northing"])
        w.writerows(gcps)
        if out is not sys.stdout:
            out.close()

    def add_gcps(self, gcp_csv_file, mobile_file):
        """
        Given a gcp file in csv format (can have Z values or different extension)
        use gdaltranslate to add the GCPs to the provided mobile file by creating
        a VRT raster
        """
        # get gcps fom gcp_csv_file
        gcps = self._read_gcp_file_csv(gcp_csv_file)
        # format gcps for gdal
        gcps = itertools.chain.from_iterable([["-gcp", *_] for _ in gcps])
        # use gdaltransform to update mobile file use VRT for wins
        mobile_vrt = Path(mobile_file).stem + "_wgcps.vrt"
        # create a vrt with the gcps
        self.cs.gdaltranslate(
            "-of",
            "VRT",
            *gcps,
            mobile_file,
            mobile_vrt,
            _out=sys.stdout,
            _err=sys.stderr,
        )
        # todo: here or as a new command would be a good place to display residuals for gcps given different transform options
        return mobile_vrt

    @staticmethod
    def warp(reference_image, mobile_vrt, out_name=None, gdal_warp_args=None, tr=1.0):
        """
        Final step in workflow, given a reference image and a mobile vrt with attached GCPs
        use gdalwarp to create a modified non-virtual file that is aligned to the reference image
        """
        if gdal_warp_args is None:
            gdal_warp_args = [
                "-overwrite",
                "-tap",
                "-multi",
                "-wo",
                "NUM_THREADS=ALL_CPUS",
                "-refine_gcps",
                "0.25, 120",
                "-order",
                3,
                "-r",
                "cubic",
                "-tr",
                tr,
                tr,
            ]
        # get reference image crs
        refimgcrs = str(
            sh.gdalsrsinfo(reference_image, "-o", "proj4")
        ).strip()  # todo: on some systems I end up with an extract space or quotes, not sure I could be mis-remembering
        # update output name
        if out_name is None:
            out_name = Path(mobile_vrt).stem + "_ref.tif"
        # let's do the time warp again
        return sh.gdalwarp(
            *gdal_warp_args,
            "-t_srs",
            refimgcrs,
            mobile_vrt,
            out_name,
            _out=sys.stdout,
            _err=sys.stderr,
        )

    def im_feeling_lucky(
        self,
        ref_img,
        mobile_image,
        *other_mobile,
        ipfindkwargs=None,
        ipmatchkwargs=None,
        gdal_warp_args=None,
    ):
        """
        Georeference an mobile dataset against a reference image.
        Do it all in one go, can take N mobile datasets but assumes the first is the mobile image.
        If unsure normalize your data ahead of time
        """
        # get the matches
        matches = list(
            self.find_matches(
                ref_img,
                mobile_image,
                ipfindkwargs=ipfindkwargs,
                ipmatchkwargs=ipmatchkwargs,
            )
        )
        # convert matches to csv
        match_csv = self.matches_to_csv(matches[0])
        # loop through all the mobile data
        import tqdm

        for i, mobile in tqdm.tqdm(enumerate([mobile_image, *other_mobile])):
            # transform matches # todo: make sure I don't overwrite anything here
            new_match_csv = self.transform_matches(
                match_csv,
                mobile_image,
                mobile,
                outname=f"{i}_{Path(ref_img).stem}__{Path(mobile).stem}.csv",
            )
            # create gcps from matches csv
            self.create_gcps(
                ref_img,
                new_match_csv,
                out_name=f"{i}_{Path(ref_img).stem}__{Path(mobile).stem}.gcps",
            )
            # add gcps to mobile file
            vrt_with_gcps = self.add_gcps(
                f"{i}_{Path(ref_img).stem}__{Path(mobile).stem}.gcps", mobile
            )
            # warp file
            self.warp(ref_img, vrt_with_gcps, out_name=gdal_warp_args)

    def get_common_matches(self, ref_left_match, ref_right_match):
        """
        returns coordinates as column row (x, y).
        rasterio xy expects row column
        """
        left_matches_cr = self._read_match_file_csv(
            ref_left_match
            if ref_left_match.endswith(".csv")
            else self.matches_to_csv(ref_left_match)
        )
        right_matches_cr = self._read_match_file_csv(
            ref_right_match
            if ref_right_match.endswith(".csv")
            else self.matches_to_csv(ref_right_match)
        )
        left_matches_cr = sorted(list(map(tuple, left_matches_cr)))
        right_matches_cr = sorted(list(map(tuple, right_matches_cr)))
        ref_left_cr = [_[0:2] for _ in left_matches_cr]
        ref_right_cr = [_[0:2] for _ in right_matches_cr]
        ref_set_left = set(ref_left_cr)
        ref_set_right = set(ref_right_cr)
        ref_common_i_left = [
            i for i, pixel in enumerate(ref_left_cr) if pixel in ref_set_right
        ]
        ref_common_i_right = [
            i for i, pixel in enumerate(ref_right_cr) if pixel in ref_set_left
        ]
        common_left = [left_matches_cr[_][2:] for _ in ref_common_i_left]
        common_right = [right_matches_cr[_][2:] for _ in ref_common_i_right]
        common_ref_left = [ref_left_cr[_] for _ in ref_common_i_left]
        common_ref_right = [ref_right_cr[_] for _ in ref_common_i_right]
        return common_ref_left, common_ref_right, common_left, common_right

    def ref_in_crs(self, common, ref_img, cr=True):
        import rasterio

        with rasterio.open(ref_img) as src:
            for _ in common:
                # rasterio xy expects row, col always
                # if coords provided as col row flip them
                yield src.xy(*(_[::-1] if cr else _))

    def get_ref_z(self, common_ref_left_crs, ref_dem):
        f = sh.gdallocationinfo.bake(ref_dem, "-valonly", "-geoloc")
        for _ in common_ref_left_crs:
            yield f(*_)

    def _small_cr_to_large_rc(self, smaller, larger, cr):
        # convert the row col index points to the CRS coordinates, then index the full res raster using the CRS points
        # to get the row col for the full resolution left/right images
        # rasterio expects row col space so flip the coordinates. I should probably use named tuples for safety
        rc = cr[::-1]
        in_crs = smaller.xy(*rc)
        row, col = larger.index(*in_crs)
        return row, col

    def make_ba_gcps(
        self,
        ref_img,
        ref_dem,
        ref_left_match,
        ref_right_match,
        left_name,
        lr_left_name,
        right_name,
        lr_right_name,
        eoid="+proj=longlat +R=3396190 +no_defs",
        out_name=None,
    ):
        import rasterio

        # get common points
        common_ref_left, common_ref_right, common_left, common_right = (
            self.get_common_matches(ref_left_match, ref_right_match)
        )
        common_ref_left_crs = list(self.ref_in_crs(common_ref_left, ref_img))
        common_ref_left_z = list(self.get_ref_z(common_ref_left_crs, ref_dem))
        # setup
        eoid_crs = pyproj.CRS(eoid)
        with rasterio.open(ref_img, "r") as ref:
            ref_crs = ref.crs
        ref_to_eoid_crs = pyproj.Transformer.from_crs(ref_crs, eoid_crs, always_xy=True)
        with rasterio.open(left_name) as left, rasterio.open(
            right_name
        ) as right, rasterio.open(lr_left_name) as lr_left, rasterio.open(
            lr_right_name
        ) as lr_right:
            # left and right are in col row space of the lowres images, and the lr and nr images have the same CRS
            common_left_full = [
                self._small_cr_to_large_rc(lr_left, left, _) for _ in common_left
            ]
            common_right_full = [
                self._small_cr_to_large_rc(lr_right, right, _) for _ in common_left
            ]
        gcps = []
        reference_gsd = round(self.cs.get_image_gsd(ref_img), 1)
        left_gsd = round(self.cs.get_image_gsd(left_name), 2)
        right_gsd = round(self.cs.get_image_gsd(right_name), 2)
        left_std = round(reference_gsd / left_gsd, 1)
        right_std = round(reference_gsd / right_gsd, 1)
        left_name = Path(left_name).name
        right_name = Path(right_name).name
        # start loop
        for i, (crs_xy, z, left_rc, right_rc) in enumerate(
            zip(
                common_ref_left_crs,
                common_ref_left_z,
                common_left_full,
                common_right_full,
            )
        ):
            # crsxy needs to be in lon lat
            lon, lat = ref_to_eoid_crs.transform(*crs_xy)
            left_row, left_col = left_rc
            right_row, right_col = right_rc
            # left/right rc might need to be flipped
            # todo: xyz stds might be too lax, could likely divide them by 3
            this_gcp = [
                i,
                lat,
                lon,
                round(float(z), 1),
                reference_gsd,
                reference_gsd,
                reference_gsd,  # gcp number, lat, lon, height, x std, y std, z std,
                left_name,
                left_col,
                left_row,
                left_std,
                left_std,  # left image, column index, row index, column std, row std,
                right_name,
                right_col,
                right_row,
                right_std,
                right_std,  # right image, column index, row index, column std, row std,
            ]
            gcps.append(this_gcp)
        print(len(gcps))
        out = open(out_name, "w") if out_name is not None else sys.stdout
        w = csv.writer(out, delimiter=" ")
        w.writerows(gcps)
        if out is not sys.stdout:
            out.close()

    def make_gcps_for_ba(
        self,
        ref_img,
        ref_dem,
        left,
        right,
        eoid="+proj=longlat +R=3396190 +no_defs",
        out_name=None,
        ipfindkwargs=None,
    ):
        """
        Given a reference image and dem, and two images for a stereopair,
         automatically create GCPs for ASP's BA by finding ip match points
         common between the reference image and the left and right images for the new pair
         and sampling the z values from the reference DEM.

         note that this will create several vrt files because we want to make normalized downsampled images
         to find a good number of matches and to save time between images of large resolution differences
        """
        if ipfindkwargs is None:
            ipfindkwargs = f"--num-threads {_threads_singleprocess} --normalize --debug-image 1 --ip-per-tile 1000"
        # normalize the data
        ref_norm = self.normalize(ref_img)
        left_norm = self.normalize(left)
        right_norm = self.normalize(right)
        # make the left/right the same gsd as the reference data
        lr_left, lr_right = list(self.match_gsds(ref_norm, left_norm, right_norm))
        # compute matches
        ref_left_match, ref_right_match = list(
            self.find_matches(ref_norm, lr_left, lr_right, ipfindkwargs=ipfindkwargs)
        )
        # make gcps for ba
        self.make_ba_gcps(
            ref_img,
            ref_dem,
            ref_left_match,
            ref_right_match,
            left,
            lr_left,
            right,
            lr_right,
            eoid=eoid,
            out_name=out_name,
        )
