```python
left = None
right = None
config = None
ref_dem = None
max_disp = None
dem_gsd = 1.0
img_gsd = 0.25
alignment_method = 'rigid'
output_path = None
downsample = None
max_ba_iterations = 200
gcps = ''
postfix = '_RED.NOPROJ.cub'
step_kwargs = {}
```

#### Assumes HiProc has already been run on two images with final products in the currect working directory

# Stereo Config file contents:

```shell
$ cat {config}
```

# Setup Steps

```python
from IPython.display import Image
from asap_stereo import asap
import math
from pathlib import Path
```

```python
default_output_dir = '~/auto_asap/hirise/'
left, right = asap.HiRISE().get_hirise_order(left, right)
if output_path == None:
    output_path = default_output_dir + f'a_{left}_{right}'
```

```shell
$ mkdir -p {output_path}
```

```shell
$ cd {output_path}
```

# Meta Data init for Scripts (step 2)

```shell
$ echo "{left}" > pair.lis
$ echo "{right}" >> pair.lis
$ cat pair.lis
```

```python
!asap
hirise
step - 2
```

# Move Files around (Step 4)

```shell
$ asap hirise step-4 --postfix {postfix}
```

# Stereo Quality Report

```python
qual_report = asap.AmesPipelineWrapper().get_stereo_quality_report(f'{left}_{right}/{left}{postfix}',
                                                                   f'{left}_{right}/{right}{postfix}')
print(qual_report)
```

# Downsample images if requested

```python
if downsample:
    true_img_gsd_left = asap.AmesPipelineWrapper().get_image_gsd(f'{left}_{right}/{left}{postfix}')
    true_img_gsd_right = asap.AmesPipelineWrapper().get_image_gsd(f'{left}_{right}/{right}{postfix}')
    # take conservative approach, pick worst image GSD
    res_gsd = max(true_img_gsd_left, true_img_gsd_right)
    # this is because rescale in ISIS does not update GSD in metadata
    asap.AmesPipelineWrapper().rescale_and_overwrite(factor=downsample, postfix=postfix)
    img_gsd = math.ceil(res_gsd) * downsample
    dem_gsd = 4 * img_gsd
    print('new img gsd', img_gsd)
    print('new dem gsd', dem_gsd)

```

# Bundle Adjust (Step 6)

```shell
$ asap hirise step-6 {gcps} --max-iterations {max_ba_iterations} {asap.kwarg_parse(step_kwargs, 'step_6')} --postfix {postfix} 2>&1 | tee -i -a ./6_bundle_adjust.log ./full_log.log
```

# Parallel Stereo p1 (Step 7)

```shell
$ asap hirise step-7 {config} {asap.kwarg_parse(step_kwargs, 'step_7')} --postfix {postfix} 2>&1 | tee -i -a ./7_parellel_stereo_1.log ./full_log.log
```

# Parallel Stereo p2 (Step 8)

```shell
$ asap hirise step-8 {config} {asap.kwarg_parse(step_kwargs, 'step_8')} --postfix {postfix} 2>&1 | tee -i -a ./8_parellel_stereo_2.log ./full_log.log
```

# Good Pixel Preview

```python
both = f'{left}_{right}'
img = f'./{both}/results_ba/{both}_ba-GoodPixelMap.tif'
out = img.replace('.tif', '.png')
```

```shell
$ gdal_translate -of PNG -co worldfile=yes {img} {out}
```

```python
Image(filename=out, width=800)
```

# Produce Previews (Step 9)

```shell
$ asap hirise step-9 --mpp {dem_gsd*2} {asap.kwarg_parse(step_kwargs, 'step_9')} --postfix {postfix} 2>&1 | tee -i -a ./9_previews.log ./full_log.log
```

## Hillshade of low res DEM

```python
both = f'{left}_{right}'
img = str(next(Path(f'./{both}/results_ba/dem/').glob(f'{both}_*DEM.tif')))
out = img.replace('.tif', '_h.png')
```

```shell
$ gdaldem hillshade {img} {out}
```

```python
Image(filename=out, width=800)
```

# Hillshade based initial transform to reference DEM

```shell
$ asap hirise pre-step-10 {ref_dem} --alignment-method {alignment_method} {asap.kwarg_parse(step_kwargs, 'pre_step_10')} --postfix {postfix} 2>&1 | tee -i -a ./10_pre_hillshade_align.log ./full_log.log
```

### Visualize the ipmatch debug image

to see how good the hill shade align was, if it's bad modify step 10 below to not use the initial transform.

```shell
$ gdal_translate -of PNG ./*/results_ba/hillshade_align/out-reference_hillshade__source_hillshade.tif ./ipd.png
```

```python
Image('./ipd.png', width=800)
```

# Point Cloud align using initial transform from hillshade (Step 10)

```shell
$ asap hirise step-10 {max_disp} {ref_dem} --initial-transform hillshade_align/out-transform.txt {asap.kwarg_parse(step_kwargs, 'step_10')} --postfix {postfix} 2>&1 | tee -i -a ./10_pc_align.log ./full_log.log
```

# Produce final aligned DEM, ortho, normalized image, error image (Step 11)

```shell
$ asap hirise step-11 --mpp {dem_gsd} {asap.kwarg_parse(step_kwargs, 'step_11')}  --postfix {postfix} 2>&1 | tee -i -a ./11_dems_orthos.log ./full_log.log
```

# Adjust to Geoid (Step 12)

```shell
$ asap hirise step-12 {asap.kwarg_parse(step_kwargs, 'step_12')} 2>&1 | tee -i -a ./12_geoid_adjustment.log ./full_log.log
```

# Make full-res Ortho image (Step 13)

```shell
$ asap hirise step-11 --mpp {img_gsd} --just-ortho True {asap.kwarg_parse(step_kwargs, 'step_13')} --postfix {postfix} 2>&1 | tee -i -a ./13_img_full_ortho.log ./full_log.log
```

## Hillshade of full res geoid adjusted DEM

```python
both = f'{left}_{right}'
img = str(next(Path(f'./{both}/results_ba/dem_align/').glob(f'{both}*DEM-adj.tif')))
out = img.replace('.tif', '_h.png')
```

```shell
$ gdaldem hillshade {img} {out}
```

```python
Image(filename=out, width=800)
```
