```markdown
## Step 11: Stereo second run (step 4 of stereo in ASP)

Same as step 6, just using the new map projected images this time.

```ipython
!asap ctx step-11 {config2} 2>&1 | tee -i -a ./6_next_level_dem.log ./full_log.log
```

## Step 7&8 again: create preview DEMs and Hillshade

We have made our second point cloud, so we should export some visuals as before.
The parameter `--folder` just specifies that we are saving things into a different directory this time around.

```ipython
!asap ctx step-7 --folder results_map_ba
```

```ipython
!asap ctx step-8 --folder results_map_ba
```

## Step 12: Get PEDR Shots for PC alignment

The final important step in the make_dem workflow is to get the MOLA PEDR data for the region we care about.
Again, our data is not completely done until it has been aligned to the MOLA topography.
If we had GCPs in the bundle adjust stage this would not be as big of an issue, but since it is relatively easy to align to MOLA we don't
need to go through the process of producing GCPs.

```ipython
!asap ctx step-12 {pedr_list}  2>&1 | tee -i -a ./7_pedr_for_pc_align.log ./full_log.log
```

## Make Final GoodPixelMap and Hillshade Previews

Nothing too surprising here, just export PNG versions of the images we care about to see the DEM at this stage of the processing.

```ipython
both = f'{left}_{right}'
img = f'./{both}/results_map_ba/{both}_ba-GoodPixelMap.tif'
out = img.replace('.tif', '.png')
```

```ipython
!gdal_translate -of PNG -co worldfile=yes {img} {out}
```

```ipython
Image(filename=out)
```

```ipython
both = f'{left}_{right}'
img = f'./{both}/results_map_ba/dem/{both}_ba_24_0-DEM-hillshade.tif'
out = img.replace('.tif', '.png')
```

```ipython
!gdal_translate -of PNG -co worldfile=yes {img} {out}
```

```ipython
Image(filename=out)
```

One additional bit here, for the MOLA data, show the PEDR2TAB template if created and the amount of PEDR data we have to align to.
If the final line is less than a few hundred we could be in a bad situation.

```ipython
!cat ./{left}_{right}/PEDR2TAB.PRM
```

```ipython
!cat ./{left}_{right}/{left}_{right}_pedr4align.csv | wc -l
```

Now that we have finished the first half of the workflow we can inspect the output products for issues before moving forwards.
If there are issues noted in the log or after a particular step, that step can be re-run with different parameters until a good solution is found.

At this point, we have a completed DEM! However, it's absolute position in space maybe off from the correct position.
Therefore, we must now perform a point cloud alignment to align our DEM with reference topography, in this case MOLA PEDR data [to correct the position of the CTX DEM](https://stereopipeline.readthedocs.io/en/latest/next_steps.html?highlight=ortho#alignment-to-point-clouds-from-a-different-source).
In older versions of ASAP, this point is the dividing line between the make_dem and align_dem pipelines.

The "maxdisp" parameter in particular deserves attention.
It is the number passed to [pc_align's --max-displacement](https://stereopipeline.readthedocs.io/en/latest/tools/pc_align.html) parameter in the Ames Stereo Pipeline.
Basically, it is the value of the distance you expect to move the CTX DEM to become aligned to your reference DEM (in this case, the PEDR data).
It is generally worth estimating this number using a GIS to sample points in both the DEM and reference file, and seeing how far away they are from each other.
But, CTX can be well behaved with ASP, so we pick a default of 500 meters which can be large enough for many situations.

## Step 13: Align the DEM to MOLA

This is the most important step in the 2nd half of the workflow as all the remaining steps are just producing final science products and visuals for the logs.
This step runs [pc_align](https://stereopipeline.readthedocs.io/en/latest/tools/pc_align.html) using the provided max displacement (aka disparity). If the
logs indicate a larger displacement was observed than the user provided value it will need to be re-run using larger values or with other advanced parameters.
If users see issues it is generally easyier to re-run the pipeline at this step repeatedly in the command line or inside the Jupyter notebook.

```ipython
!asap ctx step_13 {maxdisp} 2>&1 | tee -i -a ./8_pc_align.log ./full_log.log
```

## Step 14: Make the final CTX DEM

After the previous step everything after is simple and easy as we now have a final aligned point cloud from which DEMs and ortho images can be made.
That is all the rest of the steps do, they generate final DEMs with the geoid adjustment to produce science ready DEMs and ortho images for mapping.

```ipython
!asap ctx step_14 --mpp {demgsd}  2>&1 | tee -i -a ./9_dems_orthos.log ./full_log.log
```

## Step 15: Adjust final CTX DEM to Geoid (Areoid)

```ipython
!asap ctx step_15 2>&1 | tee -i -a ./10_geoid_adjustment.log  ./full_log.log
```

## Make the final CTX Hillshade and Orthos

```ipython
!asap ctx step_8 --folder results_map_ba --output_folder dem_align 2>&1 | tee -i -a ./11_hillshade.log ./full_log.log
```

```ipython
img = './' + str(next(Path('./').glob('./*/results_map_ba/dem_align/*_ba_align_24_0-DEM-hillshade.tif')))
out = img.replace('.tif', '.png')
```

```ipython
!gdal_translate -of PNG -co worldfile=yes {img} {out}
```

```ipython
Image(filename=out)
```

```ipython
!asap ctx step_14 --mpp {imggsd} --just_ortho True  2>&1 | tee -i -a ./12_img_full_ortho.log ./full_log.log
```
```