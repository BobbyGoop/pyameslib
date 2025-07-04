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
postfix = '_RED.cub'
gcps = ''
step_kwargs = {}
```

# Stereo Config file contents:

```python
!cat
{config}
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

```python
!mkdir - p
{output_path}
```

```python
%cd
{output_path}
```

# Download images (step 1)

```python
!asap
hirise
step - 1
{left}
{right}
2 > & 1 | tee - i - a. / 1
_download.log. / full_log.log
```

# Meta Data init for Scripts (step 2)

```python
!asap
hirise
step - 2
```

# Hiedr2mosaic (step 3)

```python
!asap
hirise
step - 3
2 > & 1 | tee - i - a. / 3
_mosaic.log. / full_log.log
```

# Move Files around (Step 4)

```python
!asap
hirise
step - 4 
```

# Stereo Quality Report

```python
qual_report = asap.AmesPipelineWrapper().get_stereo_quality_report(f'{left}_{right}/{left}_RED.cub',
                                                                   f'{left}_{right}/{right}_RED.cub')
print(qual_report)
```

# Bundle Adjust (Step 6)

```python
!asap
hirise
step - 6
{gcps} - -max - iterations
{max_ba_iterations} - -postfix
{postfix}
{asap.parse_kwargs(step_kwargs, 'step_6')}
2 > & 1 | tee - i - a. / 6
_bundle_adjust.log. / full_log.log
```

# Parallel Stereo p1 (Step 7)

```python
!asap
hirise
step - 7
{config} - -postfix
{postfix}
{asap.parse_kwargs(step_kwargs, 'step_7')}
2 > & 1 | tee - i - a. / 7
_parellel_stereo_1.log. / full_log.log
```

# Parallel Stereo p2 (Step 8)

```python
!asap
hirise
step - 8
{config} - -postfix
{postfix}
{asap.parse_kwargs(step_kwargs, 'step_8')}
2 > & 1 | tee - i - a. / 8
_parellel_stereo_2.log. / full_log.log
```

# Produce Previews (Step 9)

```python
!asap
hirise
step - 9 - -mpp
{dem_gsd * 2} - -postfix
{postfix}
{asap.parse_kwargs(step_kwargs, 'step_9')}
2 > & 1 | tee - i - a. / 9
_previews.log. / full_log.log
```

# Good Pixel Preview

```python
both = f'{left}_{right}'
img = f'./{both}/results_ba/{both}_ba-GoodPixelMap.tif'
out = img.replace('.tif', '.png')
```

```python
!gdal_translate - of
PNG - co
worldfile = yes
{img}
{out}
```

```python
Image(filename=out, width=800)
```

## Hillshade of low res DEM

```python
both = f'{left}_{right}'
img = str(next(Path(f'./{both}/results_ba/dem/').glob(f'{both}_*DEM.tif')))
out = img.replace('.tif', '_h.png')
```

```python
!gdaldem
hillshade
{img}
{out}
```

```python
Image(filename=out, width=800)
```

# Hillshade based initial transform to reference DEM

```python
!asap
hirise
pre - step - 10
{ref_dem} - -alignment - method
{alignment_method} - -postfix
{postfix}
{asap.parse_kwargs(step_kwargs, 'pre_step_10')}
2 > & 1 | tee - i - a. / 10
_pre_hillshade_align.log. / full_log.log
```

### Visualize the ipmatch debug image

to see how good the hill shade align was, if it's bad modify step 10 below to not use the initial transform.

```python
!gdal_translate - of
PNG. / * / results_ba / hillshade_align / out - reference__source.tif. / ipd.png
```

```python
Image('./ipd.png', width=800)
```

# Point Cloud align using initial transform from hillshade (Step 10)

```python
!asap
hirise
step - 10
{max_disp}
{ref_dem} - -initial - transform
hillshade_align / out - transform.txt - -postfix
{postfix}
{asap.parse_kwargs(step_kwargs, 'step_10')}
2 > & 1 | tee - i - a. / 10
_pc_align.log. / full_log.log
```

# Produce final aligned DEM, ortho, normalized image, error image (Step 11)

```shell
$ asap hirise step - 11 - -mpp {dem_gsd} {asap.kwarg_parse(step_kwargs, 'step_11')} - -postfix {postfix} 2>&1 | tee - i - a. / 11_dems_orthos.log. / full_log.log
```

# Adjust to Geoid (Step 12)

```shell
$ asap hirise step-12 {asap.kwarg_parse(step_kwargs, 'step_12')} 2 > & 1 | tee - i - a. / 12_geoid_adjustment.log. / full_log.log
```

# Make full-res Ortho image (Step 13)

```shell
$ asap hirise step-11 --mpp {img_gsd} --just -ortho True --postfix
{postfix}
{asap.kwarg_parse(step_kwargs, 'step_13')}
2 > & 1 | tee - i - a. / 13
_img_full_ortho.log. / full_log.log
```

## Hillshade of full res geoid adjusted DEM

```python
both = f'{left}_{right}'
img = str(next(Path(f'./{both}/results_ba/dem_align/').glob(f'{both}*DEM-adj.tif')))
out = img.replace('.tif', '_h.png')
```

```python
!gdaldem
hillshade
{img}
{out}
```

```python
Image(filename=out, width=800)

```
