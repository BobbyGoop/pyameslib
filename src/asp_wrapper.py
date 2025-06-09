import logging
import os.path
from collections import namedtuple

from src import moody

logging.basicConfig(level=logging.INFO)

from typing import Dict, Union
from string import Template
import sys
import logging
import re
from pathlib import Path
import math
import json
import warnings

import sh
from sh import Command
import src.moody
import pvl

from src.utils.common import _processes, _threads_multiprocess, _threads_singleprocess, \
    custom_log, convert_kwargs, silent_cd, circ_mean, rich_logger, cd, clean_kwargs, optional


class AmesPipelineWrapper:
    """
    ASAP Stereo Pipeline - Common Commands


    :ivar parallel_stereo: Команда ASP parallel_stereo — основной инструмент для
     стереообработки изображений, выполняет все этапы построения цифровых моделей рельефа.

    :ivar point2dem: Команда ASP point2dem — преобразует облако точек в цифровую модель
     рельефа (DEM) и ортофотоплан.

    :ivar pc_align: Команда ASP pc_align — выравнивает облака точек или DEM относительно
     эталонных данных (например, PEDR или другой DEM).

    :ivar dem_geoid: Команда ASP dem_geoid — корректирует значения высот DEM
     относительно геоида.

    :ivar geodiff: Команда ASP geodiff — вычисляет разницу между двумя DEM или DEM и
     CSV-файлом с высотами.

    :ivar mroctx2isis: Команда ISIS3 mroctx2isis — конвертирует исходные изображения
     в формат ISIS3.

    :ivar spiceinit: Команда ISIS3 spiceinit — добавляет SPICE-данные (геометрия, ориентация)
     к ISIS-кубам.

    :ivar spicefit: Команда ISIS3 spicefit — уточняет SPICE-данные для ISIS-кубов.

    :ivar cubreduce: Команда ISIS3 reduce — уменьшает размер изображений (кубов)
     по строкам и столбцам.

    :ivar ctxcal: Команда ISIS3 ctxcal — выполняет радиометрическую калибровку изображений CTX.

    :ivar ctxevenodd: Команда ISIS3 ctxevenodd — корректирует четные/нечетные
     строки в изображениях CTX.

    :ivar hillshade: Команда GDAL gdaldem hillshade — создает карту теней (hillshade) по DEM.

    :ivar mapproject: Команда ASP mapproject — проецирует изображения на поверхность DEM.

    :ivar ipfind: Команда ASP ipfind — находит интересные точки (interest points)
     на изображениях.

    :ivar ipmatch: Команда ASP ipmatch — сопоставляет интересные точки между изображениями.

    :ivar gdaltranslate: Команда GDAL gdal_translate — преобразует и конвертирует растровые
     изображения между форматами.

    :ivar ba: Команда ASP parallel_bundle_adjust — выполняет пакетную (bundle)
     фотограмметрическую уравниловку.
    """

    parallel_stereo: Command
    point2dem: Command
    pc_align: Command
    dem_geoid: Command
    geodiff: Command
    mroctx2isis: Command
    spiceinit: Command
    spicefit: Command
    cubreduce: Command
    ctxcal: Command
    ctxevenodd: Command
    hillshade: Command
    mapproject: Command
    ipfind: Command
    ipmatch: Command
    gdaltranslate: Command
    ba: Command

    # defaults_ps_s0 = {
    #     '--processes': _processes,
    #     '--threads-singleprocess': _threads_singleprocess,
    #     '--threads-multiprocess': _threads_multiprocess,
    #     '--entry-point': 0,
    #     '--stop-point': 1,
    #     '--bundle-adjust-prefix': 'adjust/ba'
    # }
    #
    # defaults_ps_s1 = {
    #     '--processes': _processes,
    #     '--threads-singleprocess': _threads_singleprocess,
    #     '--threads-multiprocess': _threads_multiprocess,
    #     '--entry-point': 1,
    #     '--stop-point': 2,
    #     '--bundle-adjust-prefix': 'adjust/ba'
    # }
    #
    # defaults_ps_s2 = {
    #     '--processes': _processes,
    #     '--threads-singleprocess': _threads_singleprocess,
    #     '--threads-multiprocess': _threads_multiprocess,
    #     '--entry-point': 2,
    #     '--stop-point': 3,
    #     '--bundle-adjust-prefix': 'adjust/ba'
    # }
    #
    # defaults_ps_s3 = {
    #     '--processes': _processes,
    #     '--threads-singleprocess': _threads_singleprocess,
    #     '--threads-multiprocess': _threads_multiprocess,
    #     '--entry-point': 3,
    #     '--stop-point': 4,
    #     '--bundle-adjust-prefix': 'adjust/ba'
    # }
    #
    # defaults_ps_s4 = {
    #     '--processes': _processes,
    #     '--threads-singleprocess': _threads_singleprocess,
    #     '--threads-multiprocess': _threads_multiprocess,
    #     '--entry-point': 4,
    #     '--stop-point': 5,
    #     '--bundle-adjust-prefix': 'adjust/ba'
    # }
    #
    # defaults_ps_s5 = {
    #     '--processes': _threads_singleprocess,  # use more cores for triangulation!
    #     '--threads-singleprocess': _threads_singleprocess,
    #     '--threads-multiprocess': _threads_multiprocess,
    #     '--entry-point': 5,
    #     '--bundle-adjust-prefix': 'adjust/ba'
    # }
    #
    # # defaults for first 5 (0-4 inclusive) steps parallel stereo
    # defaults_ps1 = {
    #     '--processes': _processes,
    #     '--threads-singleprocess': _threads_singleprocess,
    #     '--threads-multiprocess': _threads_multiprocess,
    #     '--stop-point': 5,
    #     '--bundle-adjust-prefix': 'adjust/ba'
    # }
    #
    # # defaults for first last step parallel stereo (triangulation)
    # defaults_ps2 = {
    #     '--processes': _threads_singleprocess,  # use more cores for triangulation!
    #     '--threads-singleprocess': _threads_singleprocess,
    #     '--threads-multiprocess': _threads_multiprocess,
    #     '--entry-point': 5,
    #     '--bundle-adjust-prefix': 'adjust/ba'
    # }

    _stereo_config = {
        '--stereo-algorithm': 'asp_mgm',
        '--processes': _processes,
        '--threads-singleprocess': _threads_singleprocess,
        '--threads-multiprocess': _threads_multiprocess,
    }

    # default eqc Iau projections, eventually replace with proj4 lookups
    projections = {
        "IAU_Mars": "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=3396190 +b=3396190 +units=m +no_defs",
        "IAU_Moon": "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs",
        "IAU_Mercury": "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=2439700 +b=2439700 +units=m +no_defs"
    }

    Stereo = namedtuple('Stereo', ['workdir', 'left', 'right'])

    def __init__(self):
        # _args = {
        #     '_out': sys.stdout,
        #     '_err': sys.stderr,
        #     '_log_msg': custom_log
        # }
        self.parallel_stereo = Command('parallel_stereo').bake(_out=sys.stdout,
                                                 _err=sys.stderr, _log_msg=custom_log)
        self.point2dem = Command('point2dem').bake('--threads', _threads_singleprocess,
                                                   _out=sys.stdout,
                                                   _err=sys.stderr, _log_msg=custom_log)
        self.point2mesh= Command('point2mesh').bake('--threads', _threads_singleprocess,
                                                   _out=sys.stdout,
                                                   _err=sys.stderr, _log_msg=custom_log)
        self.pc_align = Command('pc_align').bake('--save-inv-transform', _out=sys.stdout,
                                                 _err=sys.stderr, _log_msg=custom_log)
        self.dem_geoid = Command('dem_geoid').bake(_out=sys.stdout, _err=sys.stderr,
                                                   _log_msg=custom_log)
        self.geodiff = Command('geodiff').bake('--float', _out=sys.stdout, _err=sys.stderr,
                                               _tee=True, _log_msg=custom_log)
        self.mroctx2isis = Command('mroctx2isis').bake(_out=sys.stdout, _err=sys.stdout,
                                                       _log_msg=custom_log)
        self.spiceinit = Command('spiceinit').bake(_out=sys.stdout, _err=sys.stderr,
                                                   _log_msg=custom_log)
        self.spicefit = Command('spicefit').bake(_out=sys.stdout, _err=sys.stderr,
                                                 _log_msg=custom_log)
        self.cubreduce = Command('reduce').bake(_out=sys.stdout, _err=sys.stderr,
                                                _log_msg=custom_log)
        self.ctxcal = Command('ctxcal').bake(_out=sys.stdout, _err=sys.stderr,
                                             _log_msg=custom_log)
        self.ctxevenodd = Command('ctxevenodd').bake(_out=sys.stdout, _err=sys.stderr,
                                                     _log_msg=custom_log)
        self.hillshade = Command('gdaldem').hillshade.bake(_out=sys.stdout, _err=sys.stderr,
                                                           _log_msg=custom_log)
        self.mapproject = Command('mapproject').bake(_out=sys.stdout, _err=sys.stderr,
                                                     _log_msg=custom_log)
        self.ipfind = Command('ipfind').bake(_out=sys.stdout, _err=sys.stderr,
                                             _log_msg=custom_log)
        self.ipmatch = Command('ipmatch').bake(_out=sys.stdout, _err=sys.stderr,
                                               _log_msg=custom_log)
        self.gdaltranslate = Command('gdal_translate').bake(_out=sys.stdout, _err=sys.stderr,
                                                            _log_msg=custom_log)

        # get the help for parallel bundle adjust which changed between 3.x versions
        # pba_help = sh.parallel_bundle_adjust('--help')
        # pk = '--threads'
        # if hasattr(pba_help, '--threads-singleprocess'):
        #     pk = '--threads-singleprocess'

        self.ba = Command('parallel_bundle_adjust').bake(
            '--threads', _threads_singleprocess,
            _out=sys.stdout, _err=sys.stderr, _log_msg=custom_log
        )

        self._isd_generate = Command('isd_generate').bake(_out=sys.stdout, _err=sys.stderr,
                                                          _log_msg=custom_log)

        self.stereo_pair = None

    # @staticmethod
    # def gen_csm(self, *cubs, meta_kernal=None, max_workers=_threads_singleprocess):
    #     """
    #     Given N cub files, generate json camera models for each using ale
    #     """
    #     args = {}
    #     if meta_kernal:
    #         args['-k'] = meta_kernal
    #     cmd = sh.isd_generate('-v', *kwargs_to_args(args), '--max_workers', max_workers, *cubs,
    #                           _out=sys.stdout, _err=sys.stderr, _log_msg=custom_log)
    #     return cmd

    @property
    def workdir(self):
        return self.stereo_pair.workdir

    def _get_pair(self):
        return (self.stereo_pair.left, self.stereo_pair.right)

    def setup_pair_info(self, workdir, left, right):
        self.stereo_pair = self.Stereo(workdir, left, right)

    @staticmethod
    def cam_test(cub: str, camera: str, sample_rate: int = 1000, subpixel_offset=0.25) -> str:
        """
        """
        return sh.cam_test('--image', cub, '--cam1', cub, '--cam2', camera, '--sample-rate',
                           sample_rate, '--subpixel-offset', subpixel_offset,
                           _log_msg=custom_log)

    @staticmethod
    def get_stereo_quality_report(cub1, cub2) -> str:
        """
        Get the stereo quality report for two cub files
        The cub files must be Level1 images (Spiceinit'ed but not map-projected).

        The quality values reported by this program are based on the
        recommendations and limitations in Becker et al. (2015).  They have
        a value of one for an ideal value, between zero and one for a value
        within the acceptable limits, and less than zero (the more negative,
        the worse) if the value is beyond the acceptable limit.
        # TODO refactor into more granular bits
        :param cub1: path
        :param cub2: path
        :return:
        """
        from src.utils.stereo_quality import get_report
        report = get_report(cub1, cub2)
        return report

    @staticmethod
    def get_cam_info(img) -> Dict:
        """
        Get the camera information dictionary from ISIS using camrange

        :param img: path to image as a string
        :return: dictionary of info
        """
        wd = str(Path(img).absolute().parent)
        with silent_cd(wd):
            # currently have to cd into the directory to minify the length
            # of the file name parameter, isis3 inserts additional new lines to wrap
            # words in the terminal that will mess up isis3 to dict without management
            camrange = Command('camrange').bake(_log_msg=custom_log)
            from_path = str(Path(img).name)
            to_path = f'{str(Path(img).stem)}_camrange'
            cam_res = camrange(f'from={from_path}', f'to={to_path}')
            out_dict = pvl.load(f'{to_path}.txt')
        return out_dict

    @staticmethod
    def get_image_band_stats(img) -> dict:
        """
        :param img:
        :return:
        """
        gdalinfocmd = Command('gdalinfo').bake(_log_msg=custom_log)
        gdal_info = json.loads(str(gdalinfocmd('-json', '-stats', img)))
        return gdal_info['bands']

    @staticmethod
    def drg_to_cog(img, scale_bound: float = 0.001, gdal_options=None):
        if gdal_options is None:
            gdal_options = ["--config", "GDAL_CACHEMAX", "2000", "-co", "PREDICTOR=2", "-co",
                            "COMPRESS=ZSTD", "-co", "NUM_THREADS=ALL_CPUS", ]
        band_stats = AmesPipelineWrapper.get_image_band_stats(img)[
            0]  # assumes single band image for now
        # make output name
        out_name = Path(img).stem + '_norm.tif'
        # get bands scaling iterable, multiply by 1.001 for a little lower range
        nmin = float(band_stats["min"])
        nmax = float(band_stats["max"])
        if nmin <= 0:
            if nmin == 0:
                nmin -= 0.000001
            nmin *= (1 + scale_bound)
        else:
            nmin *= (1 - scale_bound)
        if nmax >= 0:
            if nmax == 0:
                nmax += 0.000001
            nmax *= (1 + scale_bound)
        else:
            nmax *= (1 - scale_bound)
        # run gdal translate
        return sh.gdal_translate(*gdal_options, '-of', 'COG', '-ot', 'Byte', '-scale', nmin,
                                 nmax, 1, 255, '-a_nodata', 0, img, out_name, _out=sys.stdout,
                                 _err=sys.stderr, _log_msg=custom_log)

    @staticmethod
    def get_image_gsd(img, opinion='lower') -> float:
        gdalinfocmd = Command('gdalinfo').bake('-json')
        gdal_info = json.loads(str(gdalinfocmd(img)))
        if "geoTransform" in gdal_info:
            transform = gdal_info["geoTransform"]
            res1, res2 = math.fabs(transform[1]), math.fabs(transform[-1])
        else:
            cam_info = AmesPipelineWrapper.get_cam_info(img)
            if "PixelResolution" not in cam_info:
                raise RuntimeError(
                    "Could not find pixel size for input using gdal or camrange. Check if image is valid.")
            res1, res2 = math.fabs(float(cam_info["PixelResolution"]["Lowest"])), math.fabs(
                float(cam_info["PixelResolution"]["Highest"]))
        if opinion.lower() == 'lower':
            return min(res1, res2)
        elif opinion.lower() == 'higher':
            return max(res1, res2)
        elif opinion.lower() == 'average':
            return (res1 + res2) / 2
        else:
            raise RuntimeError(
                f'Opinion {opinion} is not valid, must be "lower" or "higher" or "average".')

    @staticmethod
    def get_srs_info(img, use_eqc: str = None) -> str:
        if use_eqc:
            print(f'Using user provided projection {use_eqc}')
            return use_eqc
        try:
            # todo: depending on sh version, command may return error message in stdout/stderr
            # or empty string. if you ended up with a dem that was 2x2 this is the reason
            proj4str = str(sh.gdalsrsinfo(str(img), o='proj4'))
            if 'ERROR' in proj4str or len(proj4str) < 10:
                raise RuntimeError(f'Gdalsrsinfo failed: {proj4str}')
        except (sh.ErrorReturnCode, RuntimeError) as e:
            warnings.warn(
                f'No SRS info, falling back to use ISIS caminfo.\n exception was: {e}')
            out_dict = AmesPipelineWrapper.get_cam_info(img)
            lon = circ_mean(float(out_dict['UniversalGroundRange']['MinimumLongitude']),
                            float(out_dict['UniversalGroundRange']['MaximumLongitude']))
            lat = (float(out_dict['UniversalGroundRange']['MinimumLatitude']) + float(
                out_dict['UniversalGroundRange']['MaximumLatitude'])) / 2
            proj4str = f"+proj=ortho +lon_0={lon} +lat_0={lat} +x_0=0 +y_0=0 +a={float(out_dict['Target']['RadiusA'])} +b={float(out_dict['Target']['RadiusB'])} +units=m +no_defs"
        return str(proj4str).rstrip('\n\' ').lstrip('\'')

    @staticmethod
    def get_map_info(img, key: str, group='UniversalGroundRange') -> str:
        out_dict = AmesPipelineWrapper.get_cam_info(img)
        return out_dict[group][key]

    @staticmethod
    def parse_stereopairs():
        left, right, both = sh.cat('./stereopairs.lis').strip().split(' ')
        assert both is not None
        return left, right, both

    @staticmethod
    def create_stereopairs_lis():
        left, right, _ = sh.cat('./pair.lis').split('\n')
        with open('./stereopairs.lis', 'w') as out:
            out.write(f'{left} {right} {left}_{right}')

    @staticmethod
    def create_stereodirs_lis():
        with open('./stereodirs.lis', 'w') as out:
            _, _, left_right = AmesPipelineWrapper.parse_stereopairs()
            out.write(left_right)

    @staticmethod
    def create_stereodirs():
        Path(sh.cat('./stereodirs.lis').strip()).mkdir(exist_ok=True)

    @staticmethod
    def create_stereopair_lis():
        left, right, left_right = AmesPipelineWrapper.parse_stereopairs()
        with open(f'./{left_right}/stereopair.lis', 'w') as out:
            out.write(f'{left} {right}')

    @staticmethod
    def get_img_crs(img):
        """
        Get CRS of the image

        uses rasterio
        :param img: path to image
        :return: CRX of image
        """
        import rasterio as rio
        with rio.open(img) as i:
            return i.crs

    @staticmethod
    def get_img_bounds(img):
        """
        Get the bounds of the image

        uses rasterio
        :param img: path to image
        :return: bounds tuple
        """
        import rasterio as rio
        with rio.open(img) as i:
            return i.bounds

    def transform_bounds_and_buffer(self, img1, img2, factor=2.0):
        """
        Get bounds of img2 based on centroid of img1 surrounded by a buffer
        the size of the maximum dimension of img1 (scaled by a factor)

        ie if img1 is hirise and img2 is ctx, we find the center point of img1 in img2
        then create a bounding box that is buffered (in radius) by the height of the hirise image
        technically the buffered box would be 2x the height of the hirise which is fine

        :param img1: img to find the bounds in img2 space
        :param img2: crs we are interested in finding the expanded bounds of img1 in
        :param factor: how big we want it (radius is longest dim in img1)
        :return: xmin_img2, ymin_img2, xmax_img2, ymax_img2
        """
        from pyproj import transform
        img1_bounds = self.get_img_bounds(img1)
        img1_crs = self.get_img_crs(img1)
        img2_crs = self.get_img_crs(img2)
        # get the buffer radius
        buffer_radius = max((abs(img1_bounds.top - img1_bounds.bottom),
                             abs(img1_bounds.left - img1_bounds.right))) * factor
        # get the centroid of img1
        img1_center = (0.0, (img1_bounds.top + img1_bounds.bottom) / 2)
        # transform the centroid
        img1_center_t = transform(img1_crs, img2_crs, *img1_center)
        # use the transformed center to get new xmin ymin xmax ymax
        xmin_img2 = img1_center_t[0] - buffer_radius
        ymin_img2 = img1_center_t[1] - buffer_radius
        xmax_img2 = img1_center_t[0] + buffer_radius
        ymax_img2 = img1_center_t[1] + buffer_radius
        return xmin_img2, ymin_img2, xmax_img2, ymax_img2

    def crop_by_buffer(self, ref, src, img_out=None, factor=2.0):
        """
        use gdal warp to crop img2 by a buffer around img1

        :param ref: first image defines buffer area
        :param src: second image is cropped by buffer from first
        :param factor: factor to buffer with
        """
        xmin, ymin, xmax, ymax = self.transform_bounds_and_buffer(ref, src, factor=factor)
        img2_path = Path(src).absolute()
        if img_out == None:
            img_out = img2_path.stem + '_clipped.tif'
        return sh.gdalwarp('-te', xmin, ymin, xmax, ymax, img2_path, img_out,
                           _log_msg=custom_log)

    def check_mpp_against_true_gsd(self, path, mpp):
        """
        Get the GSD of the image, and warn if it is less than 3 * the gsd

        :param path: path to image
        :param mpp: proposed mpp for image
        """
        true_gsd = self.get_image_gsd(path)
        if mpp < true_gsd * 3:
            message = f"True image GSD is possibly too big for provided mpp value of {mpp} (compare to 3xGSD={true_gsd * 3})"
            warnings.warn(message, category=RuntimeWarning)

    @staticmethod
    def get_mpp_postfix(mpp: Union[int, float, str]) -> str:
        """
        get the mpp postfix

        :param mpp: mpp value
        :return: postfix as a string
        """
        return str(float(mpp)).replace('.', '_')

    @rich_logger
    def get_pedr_4_pcalign_w_moody(self,
                                   cub_path,
                                   proj=None,
                                   https=True,
                                   data_dir='pedr',
                                   output_prefix='pedr4align'
                                   ) -> None:
        """
        Python replacement for pedr_bin4pc_align.sh
        that uses moody and the PDS geosciences node REST API

        :param proj: optional projection override
        :param https: optional way to disable use of https
        :param cub_path: path to input file to get query geometry
        """
        cub_path = Path(cub_path).absolute()

        pedr_dir = self.stereo_pair.workdir + data_dir

        if not os.path.exists(pedr_dir):
            os.makedirs(pedr_dir)

        # out_name = cub_path.parent.name
        # cwd = cub_path.parent
        # with cd(cwd):

        out_dict = AmesPipelineWrapper.get_cam_info(cub_path)['UniversalGroundRange']
        minlon, maxlon, minlat, maxlat = out_dict['MinimumLongitude'], out_dict[
            'MaximumLongitude'], out_dict['MinimumLatitude'], out_dict['MaximumLatitude']

        # use moody to get the pedr in shape file form, we export a csv for what we need to align to
        moody.ODE(https=https).pedr(
            minlon=float(minlon),
            minlat=float(minlat),
            maxlon=float(maxlon),
            maxlat=float(maxlat),
            output_dir=pedr_dir,
            ext='shp'
        )

        shpfile = next(Path(pedr_dir).glob('*z.shp'))
        out_name = self.stereo_pair.left

        sql_query = f'SELECT Lat, Lon, Planet_Rad - 3396190.0 AS Datum_Elev, Planet_Rad, Topography FROM "{shpfile.stem}"'

        # create the minified file just for pc_align
        sh.ogr2ogr('-f', 'CSV', '-sql', sql_query, self.workdir + f"{output_prefix}.csv",
                   shpfile.absolute(), _log_msg=custom_log)

        # get projection info
        projection = self.get_srs_info(cub_path, use_eqc=proj)
        # print(projection)

        # reproject to image coordinates for some gis tools
        # todo: this fails sometimes on the projection string, a proj issue... trying again in command line seems to fix it
        sh.ogr2ogr('-t_srs', projection, '-sql', sql_query, self.workdir + f"{data_dir}/{output_prefix}.shp",
                   shpfile.absolute(), _log_msg=custom_log)

        # return f'{str(cwd)}/{out_name}_pedr4align.csv'

    def generate_csm(self, postfix='_RED.cub', camera_postfix='_RED.json'):
        """
        generate CSM models for both images
        :param postfix:
        :param camera_postfix:
        :return:
        """

        # TODO: Add meta folder

        # left, right, both = self.parse_stereopairs()
        left, right = self.stereo_pair.left, self.stereo_pair.right

        # with cd(Path.cwd() / both):
        _left, _right = f'{left}{postfix}', f'{right}{postfix}'
        _leftcam, _rightcam = f'{left}{camera_postfix}', f'{right}{camera_postfix}'

        # generate csm models
        # self.gen_csm(_left, _right)

        # Given N cub files, generate json camera models for each using ale

        args = {}
        meta_kernel = None
        if meta_kernel:
            args['-k'] = meta_kernel

        max_workers = _threads_singleprocess

        cmd = self._isd_generate(
            '-v', *convert_kwargs(args),
            '--max_workers', max_workers,
            _left, _right,
        )
        # return cmd

        # test CSMs
        print(str(self.cam_test(_left, _leftcam)))
        print(str(self.cam_test(_right, _rightcam)))
        return [(_left, _right), (_leftcam, _rightcam)]

    @rich_logger
    def bundle_adjust(self,
                      postfix='_RED.cub',
                      bundle_adjust_prefix='adjust/ba',
                      camera_postfix='.json',
                      **kwargs
                      ) -> sh.RunningCommand:
        """
        Bundle adjustment wrapper

        :param postfix: postfix of images to bundle adjust
        :param camera_postfix: postfix for cameras to use
        :param bundle_adjust_prefix: where to save out bundle adjust results
        :param kwargs: kwargs to pass to bundle_adjust
        :return: RunningCommand
        """

        # TODO: make function that attempts to find absolute paths to vargs if they are files?

        # generate the csm models first
        (_left, _right), (_leftcam, _rightcam) = self.generate_csm(postfix=postfix,
                                                                   camera_postfix=camera_postfix)

        # setup defaults
        defaults = {
            '--datum': "D_MARS",
            '--max-iterations': '100'
        }

        return self.ba(
            _left, _right, _leftcam, _rightcam,
            '-o', self.stereo_pair.workdir + bundle_adjust_prefix,
            '--save-cnet-as-csv',
            convert_kwargs(defaults | clean_kwargs(kwargs))
        )

    @rich_logger
    def stereo_asap(self,
                    entry_point: int = 0,
                    stop_point: int = 5,
                    stereo_conf: str = None,
                    cub_postfix='.lev1.eo.cub',
                    cam_postfix='.json',
                    run='results',
                    output_file_prefix='${run}/out',
                    bundle_adjust_prefix='adjust/ba',
                    ref_dem=None,
                    **kwargs):
        """
        Runs parallel_stereo command.

        Usage (From official documentation):

        ``parallel_stereo [options] <images> [<cameras>] <output_file_prefix>``

        Step 0 (Preprocessing)
        Runs stereo_pprc. Normalizes the two images and aligns them by locating interest points
        and matching them in both images. The program is designed to reject outlier interest
        points. This stage writes out the pre-aligned images and the image masks. It also computes
        the convergence angle for this stereo pair (for non-mapprojected images and with
        alignment method homography, affineepipolar, or local_epipolar).

        Step 1 (Stereo correlation)
        Runs stereo_corr. Performs correlation using various algorithms which can be specified
        via --stereo-algorithm. It writes a disparity map ending in D.tif.

        Step 2 (Blend)
        Runs stereo_blend. Blend the borders of adjacent disparity map tiles obtained during stereo
        correlation. Needed for all stereo algorithms except the classical ASP_BM when run without
        local epipolar alignment. The result is the file ending in B.tif.

        Step 3 (Sub-pixel refinement)
        Runs stereo_rfne. Performs sub-pixel correlation that refines the disparity map. Note that
        all stereo algorithms except ASP_BM already do their own refinement at step 1, however further
        refinement can happen at this step if the --subpixel-mode option is set. This produces
        a file ending in RD.tif.

        Step 4 (Outlier rejection)
        Runs stereo_fltr. Performs filtering of the disparity map and (optionally) fills in holes
        using an inpainting algorithm. It creates F.tif. Also computes GoodPixelMap.tif.

        Step 5 (Triangulation)
        Runs stereo_tri. Generates a 3D triangulated point cloud from the disparity map by
        intersecting rays traced from the cameras. The output filename ends in PC.tif.

        It is important to note that since parallel_stereo can use a lot of computational and storage
        resources, all the intermediate data up to but not including triangulation can often be reused,
        if only the cameras or camera adjustments change (for example, if the cameras got moved,
        per Section 16.54.14). Such reuse is discussed in Section 8.28.11 (in the context of stereo
        with shallow water).

        If the program failed during correlation, such as because of insufficient memory,
        it can be told to resume without recomputing the existing good partial results with the
        option --resume-at-corr.



        :param entry_point: entry point for ``parallel_stereo`` (from 0 to 4)
        :param stop_point: stop point for ``parallel_stereo`` (from 1 to 5)
        :param run: stereo run output folder prefix
        :param output_file_prefix: template string for output file prefix

        :param cub_postfix: postfix(s) to use for input images
        :param cam_postfix: postfix for cameras to use
        :param stereo_conf: stereo config file
        :param kwargs: keyword arguments for ``parallel_stereo``
        :param bundle_adjust_prefix: info about bundle adjust
        :param ref_dem: optional reference DEM to use for point cloud alignment
        """

        if stop_point < entry_point:
            raise ValueError("Stop point must be grater than entry point")

        output_file_prefix = Template(output_file_prefix).safe_substitute(
            run=self.stereo_pair.workdir + run)
        print(output_file_prefix)
        # with cd(Path.cwd() / both):
        if stereo_conf:
            kwargs['--stereo-file'] = Path(stereo_conf).absolute()

        kwargs['--entry-point'] = entry_point
        kwargs['--stop-point'] = stop_point
        kwargs['--bundle-adjust-prefix'] = self.stereo_pair.workdir + bundle_adjust_prefix

        _options = convert_kwargs(self._stereo_config | clean_kwargs(kwargs))

        left, right = self._get_pair()
        _left, _right = f'{left}{cub_postfix}', f'{right}{cub_postfix}'
        _leftcam, _rightcam = f'{left}{cam_postfix}', f'{right}{cam_postfix}'

        # ? What is refdem, there is no such argument
        # Posargs are only images, cameras and prefix
        # _posargs = posargs.split(' ')
        # *optional(_posargs),
        # *optional(refdem)
        posargs = [_left, _right, _leftcam, _rightcam, output_file_prefix]
        if ref_dem:
            posargs.append(ref_dem)
        # Don't forget to pass reference DEM if provided

        return self.parallel_stereo(
            *_options,
            *posargs,
            # _left, _right, _leftcam, _rightcam,
            # output_file_prefix,
            # ref_dem if ref_dem else ...
        )

    @rich_logger
    def point_cloud_align(self,
                          datum: str,
                          src_dem: str,
                          ref_dem: str,
                          max_displacement: float = None,
                          highest_accuracy: bool = True,
                          kind='map_ba_pedr',
                          pc_prefix='pc_align/out-PC',
                          **kwargs):
        left, right = self._get_pair()

        # Use DEM or PEDR data as reference
        # if not ref_dem:
        #     ref_dem = str(Path.cwd() / both / f'{both}_pedr4align.csv')
        # ref_dem = Path(ref_dem).absolute()

        if not max_displacement:
            # dem = next(Path(self.stereo_pair.workdir + 'dem/').glob(f'*24_0-DEM.tif'))
            # # todo implement a new command or path to do a initial NED translation with this info
            max_displacement, _, _, _ = self.estimate_max_disparity(src_dem, ref_dem)

        defaults = {
            '--num-iterations': 4000,
            '--alignment-method': 'fgr',
            '--threads': _threads_singleprocess,
            '--datum': datum,
            '--max-displacement': max_displacement,
            '--output-prefix': self.workdir + f'{pc_prefix}_{kind}'
        }

        # with cd(Path.cwd() / both / run):

        # todo allow both to be DEMs
        kwargs.pop('postfix', None)
        kwargs.pop('with_pedr', None)
        kwargs.pop('with_hillshade_align', None)

        args = convert_kwargs(defaults | clean_kwargs(kwargs))
        if str(ref_dem).endswith('.csv'):
            args.extend(['--csv-format', '1:lat 2:lon 3:height_above_datum'])

        hq = ['--highest-accuracy'] if highest_accuracy else []

        return self.pc_align(*args, *hq, src_dem, ref_dem)

    @rich_logger
    def point_to_dem(self,
                     mpp,
                     pc_suffix,
                     just_ortho=False,
                     just_dem=False,
                     use_proj=None,
                     cub_postfix='.lev1.eo.cub',
                     run='results',
                     kind='map_ba_align',
                     output_folder='dem',
                     reference_spheroid='mars',
                     **kwargs):
        left, right = self._get_pair()

        mpp_postfix = self.get_mpp_postfix(mpp)

        proj = self.get_srs_info(f'{left}{cub_postfix}',
                                 use_eqc=self.projections.get(use_proj, use_proj))
        defaults = {
            '--reference-spheroid': reference_spheroid,
            '--nodata': -32767,
            '--output-prefix': self.stereo_pair.workdir + f'{output_folder}/out-DEM_{kind}_{mpp_postfix}',
            '--dem-spacing': mpp,
            '--t_srs': proj,
        }
        post_args = []
        if just_ortho:
            post_args.append('--no-dem')
            defaults['--orthoimage'] = str(
                next(Path(self.stereo_pair.workdir + run).glob('*L.tif')).absolute()
            )
            print(defaults)
        else:
            # check the GSD against the MPP
            self.check_mpp_against_true_gsd(f'{left}{cub_postfix}', mpp)
            post_args.extend(['--errorimage'])

        # with cd(Path.cwd() / both / run):
        #     sh.mkdir(output_folder, '-p')
        #     with cd(output_folder):
        # if pc_suffix == 'PC.tif':
        #     point_cloud = next(Path.cwd().glob(f'../*{pc_suffix}')).absolute()
        # else:
        #     point_cloud = next(Path.cwd().glob(f'*{pc_suffix}')).absolute()
        point_cloud = str(
                next(Path(self.stereo_pair.workdir + run).glob(f'*{pc_suffix}')).absolute()
        )
        pre_args = convert_kwargs({**defaults, **clean_kwargs(kwargs)})
        return self.point2dem(*pre_args, str(point_cloud), *post_args)

    @rich_logger
    def point_to_mesh(self,
                     pc_suffix,
                     step_size=10,
                     texture_size=5,
                     center= True,
                     run='results',
                     output_folder='mesh',
                     **kwargs):

        post_args = []
        if center:
            post_args.append('--center')
        defaults = {
            '--point-cloud-step-size': step_size,
            '--texture-step-size': texture_size,
            '--output-prefix': self.stereo_pair.workdir + f'{output_folder}/out-MESH',
        }
        point_cloud = str(
            next(Path(self.stereo_pair.workdir + run).glob(f'*{pc_suffix}')).absolute()
        )
        pre_args = convert_kwargs({**defaults, **clean_kwargs(kwargs)})
        return self.point2mesh(*pre_args, str(point_cloud), *post_args)

    @rich_logger
    def mapproject_both(self,
                        ref_dem='D_MARS',
                        mpp=6,
                        cub_postfix='.lev1.eo.cub',
                        cam_postfix='.lev1.eo.json',
                        bundle_adjust_prefix='adjust/ba',
                        check_gsd=True,
                        **kwargs):
        """
        Mapproject the left and right images against a reference DEM

        Usage:

        ``mapproject [options] <dem> <camera-image> <camera-model> <output-image>``

        :param check_gsd: whether to check the GSD of the images against the provided mpp
        :param ref_dem: reference dem to map project using. Overwrite with a path to a DEM file
        :param mpp: target GSD
        :param cub_postfix: postfix for cub files to use
        :param cam_postfix: postfix for cameras to use
        :param bundle_adjust_prefix: where to save out bundle adjust results
        """
        left, right = self._get_pair()

        if not ref_dem:
            ref_dem = 'D_MARS'
        else:
            # todo you can map project against the datum, check if there is a suffix
            ref_path = Path(ref_dem)
            ref_dem = ref_dem if ref_path.suffix == '' else ref_path.absolute()

        # double check provided gsd
        _left, _right = f'{left}{cub_postfix}', f'{right}{cub_postfix}'
        _leftcam, _rightcam = f'{left}{cam_postfix}', f'{right}{cam_postfix}'
        # map project both images against the reference dem
        # might need to do par do here
        kwargs['--mpp'] = mpp
        ext = 'map.tif'
        if bundle_adjust_prefix:
            kwargs['--bundle-adjust-prefix'] = self.stereo_pair.workdir + bundle_adjust_prefix
            ext = f'ba.{ext}'

        if check_gsd:
            self.check_mpp_against_true_gsd(_left, mpp)
            self.check_mpp_against_true_gsd(_right, mpp)

        _options = convert_kwargs(clean_kwargs(kwargs))


        self.mapproject(ref_dem, _left, _leftcam, f'{left}.{ext}',
                        *_options)
        self.mapproject(ref_dem, _right, _rightcam, f'{right}.{ext}',
                        *_options)

    @rich_logger
    def geoid_adjust(self,
                     src_dem_sfx: str,
                     run: str,
                     output_prefix: str = "-geoid",
                     **kwargs):
        """
        Adjust DEM to geoid

        Run geoid adjustment on dem for final science ready product
        :param run:
        :param run:
        :param kwargs:
        """
        # with cd(Path.cwd() / both / run / output_folder):
        file: Path= next(Path(run).glob(f"*{src_dem_sfx}"))
        out_name = os.path.join(file.parent.absolute(), file.stem)
        args = convert_kwargs(clean_kwargs(kwargs))
        return self.dem_geoid(*args, str(file), '-o', os.path.join(file.parent.absolute(), file.stem) + output_prefix)

    @rich_logger
    def rescale_cub(self, src_file: str, factor=4, overwrite=False, dst_file=None):
        """
        rescale an ISIS3 cub file using the 'reduce' command
        given a factor, optionaly do not overwrite file

        :param src_file: path to src cub file
        :param factor: reduction factor (number [lines, samples] / factor)
        :param overwrite: if true overwrite the src file
        :param dst_file: destination file name, append `rescaled_` if not specified
        """
        c = Path(src_file)
        if not dst_file:
            dst_file = f'{c.with_name("rescaled_" + c.name)}'
        self.cubreduce(f'from={c}', f'to={dst_file}', f'sscale={factor}', f'lscale={factor}')
        if overwrite:
            sh.mv(dst_file, src_file)

    def rescale_and_overwrite(self, factor, postfix='.lev1eo.cub'):
        """
        Rescale the left and right images

        :param factor: factor to reduce each dimension by
        :param postfix: file postfix
        """
        left, right, both = self.parse_stereopairs()
        assert both is not None
        with cd(Path.cwd() / both):
            self.rescale_cub(f'{left}{postfix}', factor=factor, overwrite=True)
            self.rescale_cub(f'{right}{postfix}', factor=factor, overwrite=True)

    def get_pedr_4_pcalign_common(self, postfix, proj, https, pedr_list=None) -> str:
        if postfix.endswith('.cub'):
            warnings.warn(
                f'pedr4pcalign_common provided postfix of {postfix}, which should not end with .cub! Removing for you..')
            postfix = postfix[:-4]
        left, right, both = self.parse_stereopairs()
        with cd(Path.cwd() / both):
            res = self.get_pedr_4_pcalign_w_moody(f'{left}{postfix}', proj=proj, https=https)
            return res

    def get_geo_diff(self, ref_dem, src_dem=None):
        left, right, both = self.parse_stereopairs()
        ref_dem = Path(ref_dem).absolute()
        with cd(Path.cwd() / both):
            # todo reproject to match srs exactly
            args = []
            if not src_dem:
                src_dem = next(Path.cwd().glob('*_pedr4align.csv'))
            src_dem = str(src_dem)
            if src_dem.endswith('.csv'):
                args.extend(['--csv-format', '1:lat 2:lon 3:height_above_datum'])
            args.extend([ref_dem, src_dem])
            if not src_dem.endswith('.csv'):
                args.extend(['-o', 'geodiff/o'])
            res = self.geodiff(*args)
            if src_dem.endswith('.csv'):
                # geodiff stats in std out for CSV
                res = str(res).splitlines()
                res = {k.strip(): v.strip() for k, v in
                       [_ for _ in (l.split(':') for l in res) if len(_) == 2]}
            else:
                # if both are dems I need to use gdalinfo for diff
                stats = self.get_image_band_stats('./geodiff/o-diff.tif')
                if isinstance(stats, list):
                    stats = stats[0]
                assert isinstance(stats, dict)
                res = {
                    'Max difference': stats["maximum"],
                    'Min difference': stats["minimum"],
                    'Mean difference': stats["mean"],
                    'StdDev of difference': stats['stdDev'],
                    'Median difference': stats["mean"],
                    # yes I know this isn't correct but gdal doens't compute this for us
                }
            for k, v in res.items():
                try:
                    res[k] = float(v)
                except ValueError:
                    try:
                        res[k] = float(''.join(re.findall(r'-?\d+\.?\d+',
                                                          v)))  # this won't grab all floats like Nans or si notation
                    except ValueError:
                        res[k] = 0.0
            return res

    def estimate_max_disparity(self, ref_dem, src_dem=None):
        """
        Estimate the absolute value of the maximum observed displacement
        between two point clouds, and the standard deviation of the differences

        if not applying an initial transform to pc_align, use the max_d value
        if expecting to apply a transform first and you are
        interested in the maximum displacement after an initial transform, then
        use the std_d returned (likely 3X it)
        """
        vals = self.get_geo_diff(ref_dem, src_dem)
        max_d = float(vals['Max difference'])
        min_d = float(vals['Min difference'])
        std_d = float(vals['StdDev of difference'])
        absmax_d = max(abs(max_d), abs(min_d))
        return absmax_d, max_d, min_d, std_d

    def estimate_median_disparity(self, ref_dem, src_dem=None):
        vals = self.get_geo_diff(ref_dem, src_dem)
        med_d = float(vals['Median difference'])
        return med_d, abs(med_d)

    def compute_footprints(self, *imgs):
        """
        for each footprint generate a vector footprint
        :param imgs: gdal rasters with nodata defined
        :return:
        """
        import tqdm
        poly = sh.Command('gdal_polygonize.py').bake(_log_msg=custom_log)
        for img in tqdm.tqdm(imgs):
            md = json.loads(str(sh.gdalinfo(img, '-json', _log_msg=custom_log)))
            if not 'noDataValue' in md['bands'][0]:
                print('no noDataValue in image: ', img)
                continue
            # downsample to 10% size
            ds_out_name = Path(img).stem + f'_ds.vrt'
            _ = sh.gdal_translate(img, ds_out_name, '-of', 'vrt', '-outsize', '10%', '10%',
                                  '-r', 'cubic', _log_msg=custom_log)
            # scale to binary
            eb_out_name = Path(img).stem + f'_eb.vrt'
            _ = sh.gdal_translate(ds_out_name, eb_out_name, '-of', 'vrt', '-scale', '-ot',
                                  'byte', _log_msg=custom_log)
            # scale to mask
            vp_out_name = Path(img).stem + f'_vp.vrt'
            _ = sh.gdal_translate(eb_out_name, vp_out_name, '-of', 'vrt', '-scale', '1', '255',
                                  '100', '100', _log_msg=custom_log)
            # make polygon
            g_out_name = Path(img).stem + f'_footprint.geojson'
            _ = poly('-of', 'geojson', '-8', vp_out_name, g_out_name)
            # cleanup intermediate products
            Path(ds_out_name).unlink(missing_ok=True)
            Path(eb_out_name).unlink(missing_ok=True)
            Path(vp_out_name).unlink(missing_ok=True)
