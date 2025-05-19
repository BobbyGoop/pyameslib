#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __BEGIN_LICENSE__
#  Copyright (c) 2009-2013, United States Government as represented by the
#  Administrator of the National Aeronautics and Space Administration. All
#  rights reserved.
#
#  The NGT platform is licensed under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance with the
#  License. You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# __END_LICENSE__

import optparse
import os
import sys
import time


def manual(option, opt, value, parser):
    print(parser.usage, file=sys.stderr)
    print(
        "Generates a stereo DEM from two LRONAC pairs using SBA and LOLA for increased accuracy.",
        file=sys.stderr
    )
    sys.exit()

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

#--------------------------------------------------------------------------------

def replace_extension_and_folder(input_path, output_folder, new_extension):
    new_ext = os.path.splitext(input_path)[0] + new_extension
    return os.path.join(output_folder, os.path.basename(new_ext))

def prepare_image(input_path, work_dir, keep):
    """Prepare a single CTX image for processing"""

    # Set up paths
    cub_path = replace_extension_and_folder(input_path, work_dir, '.cub')
    cal_path = replace_extension_and_folder(input_path, work_dir, '.cal.cub')

    # Convert to ISIS format
    cmd = f"mroctx2isis from={input_path} to={cub_path}"
    os.system(cmd)

    # Init Spice data
    cmd = f"spiceinit from={cub_path}"
    os.system(cmd)

    # Apply image correction
    cmd = f"ctxcal from={cub_path} to={cal_path}"
    os.system(cmd)

    # you can also optionally run ctxevenodd on the cal.cub files, if needed

    if not keep:
        os.remove(cub_path)

    return cal_path

def main():
    print('#' * 81)
    print("Running processCtxPair.py")

    try:
        usage = (
            "usage: processCtxPair.py <left image> <right image> <output prefix> "
            "[--workDir <folder>][--keep][--manual]\n  "
        )
        parser = optparse.OptionParser(usage=usage)

        parser.set_defaults(keep=False)

        parser.add_option("--workDir", dest="work_dir", help="Folder to place intermediate files in")
        parser.add_option("--manual", action="callback", callback=manual, help="Read the manual.")
        parser.add_option("--keep", action="store_true", dest="keep", help="Do not delete the temporary files.")

        (options, args) = parser.parse_args()

        if len(args) < 3:
            parser.error('Missing required input!')
        options.left_path = args[0]
        options.right_path = args[1]
        options.output_prefix = args[2]

        if not options.work_dir:
            options.work_dir = os.path.dirname(options.output_prefix)

        start_time = time.time()

        # Do individual input image preparations
        left_cal_path = prepare_image(options.left_path, options.work_dir, options.keep)
        right_cal_path = prepare_image(options.right_path, options.work_dir, options.keep)

        # Do joint preparation
        cmd = f"cam2map4stereo.py {left_cal_path} {right_cal_path}"
        os.system(cmd)
        left_map_path = replace_extension_and_folder(options.left_path, options.work_dir, '.map.cub')
        right_map_path = replace_extension_and_folder(options.right_path, options.work_dir, '.map.cub')

        # Final stereo call
        cmd = (
            f"parallel_stereo.py {left_map_path} {right_map_path} {options.output_prefix} "
            "--alignment affineepipolar --subpixel-mode 3 --corr-timeout 400 "
            "--filter-mode 1 --subpixel-max-levels 0"
        )
        os.system(cmd)

        # Clean up temporary files
        if not options.keep:
            os.remove(left_cal_path)
            os.remove(right_cal_path)
            os.remove(left_map_path)
            os.remove(right_map_path)

        end_time = time.time()

        print(f"Finished in {end_time - start_time} seconds.")
        print('#' * 81)
        return 0

    except Usage as err:
        print(err)
        print(err.msg, file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())