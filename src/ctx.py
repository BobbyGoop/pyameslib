import logging
from importlib.resources import as_file, files
from pathlib import Path
from typing import Optional

from src.asp_wrapper import CommonSteps
from src.utils.common import rich_logger, par_do, cd

logging.basicConfig(level=logging.INFO)

import sh
# import moody
import papermill as pm
from src.moody import ODE

class CTX(object):
    r"""
    ASAP Stereo Pipeline - CTX workflow

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
        self.cs = CommonSteps()
        self.https = https
        self.datum = datum
        # if proj is not none, get the corresponding proj or else override with proj,
        # otherwise it's a none so remain a none
        self.proj = self.cs.projections.get(proj, proj)

    def get_first_pass_refdem(self, run='results_ba') -> str:
        left, right, both = self.cs.parse_stereopairs()
        refdem = Path.cwd() / both / run / 'dem' / f'{both}_ba_100_0-DEM.tif'
        return str(refdem)

    def get_full_ctx_id(self, pid):
        res = str(ODE(self.https).get_ctx_meta_by_key(pid, 'ProductURL'))
        return res.split('=')[1].split('&')[0]

    def get_ctx_emission_angle(self, pid):
        return float(ODE(self.https).get_ctx_meta_by_key(pid, 'Emission_angle'))

    def get_ctx_order(self, one, two):
        print('Calculating emission angles with provided PIDs...')
        em_one = self.get_ctx_emission_angle(one)
        em_two = self.get_ctx_emission_angle(two)
        if em_one <= em_two:
            return one, two
        else:
            return two, one

    def generate_ctx_pair_list(self, one, two):
        order = self.get_ctx_order(one, two)
        full_ids = [self.get_full_ctx_id(pid) for pid in order]
        with open('pair.lis', 'w', encoding='utf') as o:
            for pid in full_ids:
                o.write(pid)
                o.write('\n')

    @staticmethod
    def notebook_pipeline_make_dem(left: str, right: str, config1: str, pedr_list: str = None,
                                   downsample: int = None, working_dir='./',
                                   config2: Optional[str] = None, dem_gsd=24.0, img_gsd=6.0,
                                   max_disp=None, step_kwargs=None,
                                   out_notebook=None, **kwargs):
        """
        First step in CTX DEM pipeline that uses papermill to persist log

        this command does most of the work, so it is long running!
        I recommend strongly to use nohup with this command

        :param out_notebook: output notebook log file name, defaults to log_asap_notebook_pipeline_make_dem.ipynb
        :param config2: ASP config file to use for second processing pass
        :param working_dir: Where to execute the processing, defaults to current directory
        :param config1: ASP config file to use for first processing pass
        :param pedr_list: Path to PEDR files, defaults to None to use ODE Rest API
        :param left: First image id
        :param right: Second image id
        :param max_disp: Maximum expected displacement in meters, use None to determine it automatically
        :param step_kwargs: Arbitrary dict of kwargs for steps following {'step_#' : {'key': 'value}}
        :param downsample: Factor to downsample images for faster production
        :param dem_gsd: desired GSD of output DEMs (4x image GSD)
        :param img_gsd: desired GSD of output ortho images
        :param kwargs: kwargs for papermill
        """
        if not out_notebook:
            out_notebook = f'{working_dir}/log_asap_notebook_pipeline_make_dem.ipynb'
        with as_file(files('pyameslib').joinpath('asap_ctx_workflow.ipynb')) as src:
            pm.execute_notebook(
                src,
                out_notebook,
                parameters={
                    'left': left,
                    'right': right,
                    'peder_list': pedr_list,
                    'config1': config1,
                    'config2': config2,
                    'output_path': working_dir,
                    'max_disp': max_disp,
                    'dem_gsd': dem_gsd,
                    'img_gsd': img_gsd,
                    'downsample': downsample,
                    'step_kwargs': step_kwargs
                },
                request_save_on_cell_execute=True,
                **kwargs
            )

    @rich_logger
    def step_1(self, one: str, two: str, cwd: Optional[str] = None) -> None:
        """
        Download CTX EDRs from the PDS

        :param one: first CTX image id
        :param two: second CTX image id
        :param cwd:
        """
        with cd(cwd):
            self.generate_ctx_pair_list(one, two)
            # download files
            ODE(self.https).ctx_edr(one)
            ODE(self.https).ctx_edr(two)

    @rich_logger
    def step_2(self, with_web=False):
        """
        ISIS3 CTX preprocessing, replaces ctxedr2lev1eo.sh

        :param with_web: if true attempt to use webservices for SPICE kernel data
        """
        imgs = [*Path.cwd().glob('*.IMG'), *Path.cwd().glob('*.img')]
        par_do(self.cs.mroctx2isis, [f'from={i.name} to={i.stem}.cub' for i in imgs])
        cubs = list(Path.cwd().glob('*.cub'))
        par_do(self.cs.spiceinit,
               [f'from={c.name}{" web=yes" if with_web else ""}' for c in cubs])
        par_do(self.cs.spicefit, [f'from={c.name}' for c in cubs])
        par_do(self.cs.ctxcal, [f'from={c.name} to={c.stem}.lev1.cub' for c in cubs])
        for cub in cubs:
            cub.unlink()
        lev1cubs = list(Path.cwd().glob('*.lev1.cub'))
        par_do(self.cs.ctxevenodd, [f'from={c.name} to={c.stem}eo.cub' for c in lev1cubs])
        for lc in lev1cubs:
            lc.unlink()

    @rich_logger
    def step_3(self):
        """
        Create various processing files for future steps
        # todo: deduplicate with hirise side
        """
        self.cs.create_stereopairs_lis()
        self.cs.create_stereodirs_lis()
        self.cs.create_stereodirs()
        self.cs.create_stereopair_lis()
        # copy the cub files into the both directory
        _, _, both = self.cs.parse_stereopairs()
        return sh.mv('-n', sh.glob('./*.cub'), f'./{both}/')

    @rich_logger
    def step_4(self, *vargs, bundle_adjust_prefix='adjust/ba', postfix='.lev1eo.cub',
               camera_postfix='.lev1eo.json', **kwargs) -> sh.RunningCommand:
        """
        Bundle Adjust CTX

        Run bundle adjustment on the CTX map projected data

        :param vargs: variable length additional positional arguments to pass to bundle adjust
        :param bundle_adjust_prefix: prefix for bundle adjust output
        :param postfix: postfix for cub files to use
        :param camera_postfix: postfix for cameras
        """
        return self.cs.bundle_adjust(*vargs, postfix=postfix, camera_postfix=camera_postfix,
                                     bundle_adjust_prefix=bundle_adjust_prefix, **kwargs)

    @rich_logger
    def step_5(self, stereo_conf, posargs='', postfix='.lev1eo.cub',
               camera_postfix='.lev1eo.json', **kwargs):
        """
        Parallel Stereo Part 1

        Run first part of parallel_stereo asp_ctx_lev1eo2dem.sh

        :param postfix: postfix for cub files to use
        :param camera_postfix: postfix for cameras  # TODO: use .adjusted_state.json?
        """
        return self.cs.stereo_asap(stereo_conf, postfix=postfix, camera_postfix=camera_postfix,
                                   posargs=posargs, **{**self.cs.defaults_ps1, **kwargs})

    @rich_logger
    def step_6(self, stereo_conf, posargs='', postfix='.lev1eo.cub',
               camera_postfix='.lev1eo.json', **kwargs):
        """
        Parallel Stereo Part 2

        Run second part of parallel_stereo, asp_ctx_lev1eo2dem.sh stereo is completed after this step

        :param postfix: postfix for cub files to use
        :param camera_postfix: postfix for cameras  # TODO: use .adjusted_state.json?
        """
        return self.cs.stereo_asap(stereo_conf, postfix=postfix, camera_postfix=camera_postfix,
                                   posargs=posargs, **{**self.cs.defaults_ps2, **kwargs})

    @rich_logger
    def step_7(self, mpp=24, just_ortho=False, run='results_ba', postfix='.lev1eo.cub',
               **kwargs):
        """
        Produce preview DEMs/Orthos

        Produce dem from point cloud, by default 24mpp for ctx for max-disparity estimation

        :param run: folder for results
        :param just_ortho: set to True if you only want the ortho image, else make dem and error image
        :param mpp: resolution in meters per pixel
        :param postfix: postfix for cub files to use
        """
        return self.cs.point_to_dem(mpp, 'PC.tif',
                                    just_ortho=just_ortho,
                                    postfix=postfix,
                                    run=run,
                                    kind='ba',
                                    use_proj=self.proj,
                                    **kwargs)

    @rich_logger
    def step_8(self, run='results_ba', output_folder='dem'):
        """
        hillshade First step in asp_ctx_step2_map2dem script

        :param output_folder:
        :param run:
        """
        left, right, both = self.cs.parse_stereopairs()
        with cd(Path.cwd() / both / run / output_folder):
            dem = next(Path.cwd().glob('*DEM.tif'))
            self.cs.hillshade(dem.name, f'./{dem.stem}-hillshade.tif')

    @rich_logger
    def step_9(self, refdem=None, mpp=6, run='results_ba', postfix='.lev1eo.cub',
               camera_postfix='.lev1eo.json'):
        """
        Mapproject the left and right ctx images against the reference DEM

        :param run: name of run
        :param refdem: reference dem to map project using
        :param mpp: target GSD
        :param postfix: postfix for cub files to use
        :param camera_postfix: postfix for cameras to use
        """
        left, right, both = self.cs.parse_stereopairs()
        if not refdem:
            refdem = Path.cwd() / both / run / 'dem' / f'{both}_ba_100_0-DEM.tif'
        else:
            refdem = Path(refdem).absolute()
        with cd(Path.cwd() / both):
            # double check provided gsd
            _left, _right = f'{left}{postfix}', f'{right}{postfix}'
            _leftcam, _rightcam = f'{left}{camera_postfix}', f'{right}{camera_postfix}'
            self.cs.check_mpp_against_true_gsd(_left, mpp)
            self.cs.check_mpp_against_true_gsd(_right, mpp)
            # map project both ctx images against the reference dem
            # might need to do par do here
            self.cs.mapproject(refdem, _left, _leftcam, f'{left}.ba.map.tif', '--mpp', mpp,
                               '--bundle-adjust-prefix', 'adjust/ba')
            self.cs.mapproject(refdem, _right, _rightcam, f'{right}.ba.map.tif', '--mpp', mpp,
                               '--bundle-adjust-prefix', 'adjust/ba')

    @rich_logger
    def step_10(self, stereo_conf, refdem=None, posargs='', postfix='.ba.map.tif',
                camera_postfix='.lev1eo.json', **kwargs):
        """
        Second stereo first step

        :param stereo_conf:
        :param refdem: path to reference DEM or PEDR csv file
        :param posargs: additional positional args
        :param postfix: postfix for files to use
        :param camera_postfix: postfix for cameras to use
        :param kwargs:
        """
        refdem = str(Path(self.get_first_pass_refdem() if not refdem else refdem).absolute())
        return self.cs.stereo_asap(stereo_conf=stereo_conf, refdem=refdem, postfix=postfix,
                                   camera_postfix=camera_postfix, run='results_map_ba',
                                   posargs=posargs, **{**self.cs.defaults_ps1, **kwargs})

    @rich_logger
    def step_11(self, stereo_conf, refdem=None, posargs='', postfix='.ba.map.tif',
                camera_postfix='.lev1eo.json', **kwargs):
        """
        Second stereo second step

        :param stereo_conf:
        :param refdem: path to reference DEM or PEDR csv file
        :param posargs: additional positional args
        :param postfix: postfix for files to use
        :param camera_postfix: postfix for cameras to use
        :param kwargs:
        """
        refdem = str(Path(self.get_first_pass_refdem() if not refdem else refdem).absolute())
        return self.cs.stereo_asap(stereo_conf=stereo_conf, refdem=refdem, postfix=postfix,
                                   camera_postfix=camera_postfix, run='results_map_ba',
                                   posargs=posargs, **{**self.cs.defaults_ps2, **kwargs})

    @rich_logger
    def step_12(self, pedr_list=None, postfix='.lev1eo'):
        """
        Get MOLA PEDR data to align the CTX DEM to

        :param postfix: postfix for file, minus extension
        :param pedr_list: path local PEDR file list, default None to use REST API
        """
        self.cs.get_pedr_4_pcalign_common(postfix, self.proj, self.https, pedr_list=pedr_list)

    @rich_logger
    def step_13(self, run='results_map_ba', maxd: float = None, refdem=None,
                highest_accuracy=True, **kwargs):
        """
        PC Align CTX

        Run pc_align using provided max disparity and reference dem
        optionally accept an initial transform via kwargs

        :param run: folder used for this processing run
        :param highest_accuracy: Use the maximum accuracy mode
        :param maxd: Maximum expected displacement in meters
        :param refdem: path to pedr csv file or reference DEM/PC, if not provided assume pedr4align.csv is available
        :param kwargs:
        """
        return self.cs.point_cloud_align(self.datum, maxd=maxd, refdem=refdem,
                                         highest_accuracy=highest_accuracy, run=run,
                                         kind='map_ba_align', **kwargs)

    @rich_logger
    def step_14(self, mpp=24.0, just_ortho=False, run='results_map_ba',
                output_folder='dem_align', postfix='.lev1eo.cub', **kwargs):
        """
        Produce final DEMs/Orthos

        Run point2dem on the aligned output to produce final science ready products

        :param run: folder used for this processing run
        :param mpp:
        :param just_ortho:
        :param output_folder:
        :param postfix: postfix for cub files to use
        :param kwargs:
        """
        return self.cs.point_to_dem(mpp,
                                    'trans_reference.tif',
                                    just_ortho=just_ortho,
                                    postfix=postfix,
                                    run=run,
                                    kind='map_ba_align',
                                    use_proj=self.proj,
                                    output_folder=output_folder,
                                    **kwargs)

    @rich_logger
    def step_15(self, run='results_map_ba', output_folder='dem_align', **kwargs):
        """
        Adjust DEM to geoid

        Run geoid adjustment on dem for final science ready product
        :param run: folder used for this processing run
        :param output_folder:
        :param kwargs:
        """
        return self.cs.geoid_adjust(run=run, output_folder=output_folder, **kwargs)