from itertools import chain
import fire
import requests
import sys
from tqdm import tqdm
from contextlib import closing
from typing import Optional
import shutil
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import chain


def query_params(params, key, def_value, short_hand=None):
    """
    updates params dict to use
    :param params:
    :param key:
    :param def_value:
    :param short_hand:
    :return:
    """
    if key not in params and short_hand:
        # value is associated with shorthand, move to key
        params[key] = params.get(short_hand, def_value)
        del params[short_hand]
    elif key not in params and not short_hand:
        params[key] = def_value
    elif key in params:
        # key is there, also possibly shorthand
        # assume def value at this point is not needed
        if short_hand in params:
            del params[short_hand]
    return params


def query_gds(gds_url, query):
    with closing(requests.get(gds_url, params=query)) as r:
        if r.ok:
            response = r.json()
            products_check = response["GDSResults"]
            if products_check["Status"] != "Success":
                raise RuntimeError(
                    f"Error, some issue with the query: {r.url}", products_check
                )
            else:
                return products_check
        else:
            raise RuntimeError(
                "Error with query at url: {} with code: {}".format(
                    gds_url, r.status_code
                )
            )


def query_ode(ode_url, query):
    req = requests.get(ode_url, params=query)

    if req.ok:
        response = req.json()
        products_check = response["ODEResults"]["Products"]
        if products_check == "No Products Found":
            print("Error, PID not found by ODE")
            print(req.url)
            sys.exit(1)
        else:
            return products_check["Product"]
    else:
        print(
            "Error with query at url: {} with code: {}".format(
                ode_url, req.status_code
            )
        )
        sys.exit(1)


def download_edr_img_files_par(
        products, https: bool = True, chunk_size: int = 1024 * 1024
):
    edr_products = list(
        chain.from_iterable([_["Product_files"]["Product_file"] for _ in products])
    )
    edr_files = [x for x in edr_products if x["URL"].endswith(".IMG")]
    # fix lroc urls
    for x in edr_files:
        if "www.lroc.asu.edu" in x["URL"]:
            x["URL"] = x["URL"].replace("www.lroc.asu.edu", "pds.lroc.asu.edu")
    urls = [_["URL"] for _ in edr_files]
    filenames = [_["FileName"] for _ in edr_files]
    with Pool(cpu_count()) as pool:
        get = partial(download_file, chunk_size=chunk_size)
        pool.starmap(get, list(zip(urls, filenames)))


def download_file(url, output_path, product_name, chunk_size):
    url = url.replace(
        "pds-imaging.jpl.nasa.gov/data/", "planetarydata.jpl.nasa.gov/img/data/"
    )
    full_path = os.path.join(output_path, product_name)

    if os.path.exists(full_path):
        print(f"File {product_name} already exists, skipping download.")
        return

    with open(full_path, "wb", chunk_size) as output:
        with closing(requests.get(url, stream=True, allow_redirects=True)) as r:
            for chunk in tqdm(
                    r.iter_content(chunk_size), desc=f"Downloading {product_name}"
            ):
                if chunk:
                    output.write(chunk)
                    output.flush()
            r.close()
        output.flush()
    if str(product_name).endswith(".zip"):
        shutil.unpack_archive(product_name)
        if os.path.exists(product_name):
            os.remove(product_name)