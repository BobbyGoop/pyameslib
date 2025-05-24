"""
The MIT License (MIT)

Copyright (c) [2017-2021] [Andrew Annex]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import sys
from typing import Optional

import fire

from src.utils.img import (
    query_ode,
    download_file,
    download_edr_img_files_par,
    query_gds,
    query_params
)


class ODE(object):
    """class to hold ode downloading commands"""

    def __init__(self, https=True, debug=False):
        self.https = https
        if https:
            self.ode_url = "https://oderest.rsl.wustl.edu/live2"
            self.gds_url = "https://oderest.rsl.wustl.edu/livegds"
        else:
            raise EnvironmentError('Downloading with HTTP is not supported')
            # self.ode_url = "http://oderest.rsl.wustl.edu/live2"
            # self.gds_url = "http://oderest.rsl.wustl.edu/livegds"

    def download_ctx_edr(self, pid, output_path, https: bool = True, chunk_size: int = 1024 * 1024):
        """
        Download a CTX EDR .IMG file to the CWD.

        pid: Exact product ID of the CTX EDR
        chunk_size: Chunk size in bytes to use in download
        """

        query = {
            "target": "mars",
            "query": "product",
            "results": "f",
            "output": "j",
            "pt": "EDR",
            "iid": "CTX",
            "ihid": "MRO",
            "productid": pid,
        }

        # Query the ODE
        product = query_ode(self.ode_url, query)

        # Validate query results with conditions for this particular query
        if isinstance(product, list):
            raise ValueError("Too many products selected for in query, Make PID more specific")

        edr_products = product["Product_files"]["Product_file"]
        edr_files = [x for x in edr_products if x["URL"].endswith(".IMG")]

        if len(edr_files) > 1:
            raise ValueError("Found more than one EDR file, make PID more specific")

        img_file = edr_files.pop()

        # fix lroc urls
        if "www.lroc.asu.edu" in img_file["URL"]:
            img_file["URL"] = img_file["URL"].replace("www.lroc.asu.edu", "pds.lroc.asu.edu")

        url = url_https(img_file["URL"]) if https else img_file["URL"]

        filename = img_file["FileName"]

        # make download request
        download_file(url, output_path, filename, chunk_size)

        return os.path.join(output_path, filename)

    def download_hirise_edr(self, pid, chunk_size=1024 * 1024):
        """
        Download a HiRISE EDR set of .IMG files to the CWD

        You must know the full id to specifiy the filter to use, ie:
        PSP_XXXXXX_YYYY         will download every EDR IMG file available
        PSP_XXXXXX_YYYY_R       will download every EDR RED filter IMG file
        PSP_XXXXXX_YYYY_BG12_0  will download only the BG12_0

        As a wild card is auto applied to the end of the provided pid

        pid: product ID of the HiRISE EDR, partial IDs ok
        chunk_size: Chunk size in bytes to use in download
        """
        productid = "{}*".format(pid)

        query = {
            "target": "mars",
            "query": "product",
            "results": "f",
            "output": "j",
            "pt": "EDR",
            "iid": "HiRISE",
            "ihid": "MRO",
            "productid": productid,
        }

        # Query the ODE
        products = query_ode(self.ode_url, query)
        # Validate query results with conditions for this particular query
        if len(products) > 30:
            print(
                "Error: Too many products selected for in query, Make PID more specific"
            )
            sys.exit(1)
        if not isinstance(products, list):
            print("Error: Too few responses from server to be a full HiRISE EDR, ")
        else:
            # proceed to download
            download_edr_img_files_par(products, self.https, chunk_size)

    def lrocnac_edr(self, pid, chunk_size=1024 * 1024):
        """
        Download a LROC NAC EDR set of .IMG files to the CWD

        As a wild card is auto applied to the end of the provided pid

        pid: product ID of the LROC EDR, partial IDs ok
        chunk_size: Chunk size in bytes to use in download
        """
        productid = "{}*".format(pid)

        query = {
            "target": "moon",
            "query": "product",
            "results": "f",
            "output": "j",
            "pt": "EDRNAC4",
            "iid": "LROC",
            "ihid": "LRO",
            "productid": productid,
        }

        # Query the ODE
        products = query_ode(self.ode_url, query)
        # Validate query results with conditions for this particular query
        if len(products) > 30:
            print(
                "Error: Too many products selected for in query, Make PID more specific"
            )
            sys.exit(1)
        if not isinstance(products, list):
            print("Error: Too few responses from server to be a full HiRISE EDR, ")
        else:
            # proceed to download
            download_edr_img_files_par(products, self.https, chunk_size)

    def pedr(
            self,
            minlon: float,
            minlat: float,
            maxlon: float,
            maxlat: float,
            wkt_footprint: Optional[str] = None,
            ext: str = "csv",
            **kwargs,
    ):
        """
        Get the mola pedr csv/shp file for the query bounds
        :param ext:
        :param minlon: minimum longitude (western most longitude)
        :param minlat: minimum latitude  (southern most latitude)
        :param maxlon: maximum longitude (eastern most longitude)
        :param maxlat: maximum latitude  (northern most latitude)
        :param wkt_footprint: Optional WKT footprint to further filter out points
        :return:
        """
        if minlon < 0 or maxlon < 0:
            # convert -180 to 180 to 0 to 360
            minlon += 180.0
            maxlon += 180.0
        assert 0 <= minlon <= 360
        assert 0 <= maxlon <= 360
        assert minlon < maxlon and minlat < maxlat
        # default is csv
        rt = "s" if ext == "shp" else "v"
        query = {
            "query": "molapedr",
            "results": rt,
            "output": "J",
            "minlat": str(minlat),
            "maxlat": str(maxlat),
            "westernlon": str(minlon),
            "easternlon": str(maxlon),
            "zipclean": "t",
            **kwargs,
        }
        if wkt_footprint:
            query["footprint"] = f"{wkt_footprint}"
        # Query the ODEq
        response = query_gds(self.gds_url, query)
        # get the ResultFile, it seems ResultFile has the same number of contents as Number Files
        resultfile = response["ResultFiles"]["ResultFile"]
        if isinstance(resultfile, dict):
            resultfile = [resultfile]
        for f in resultfile:
            fname = str(f["URL"].split("/")[-1])
            fname = fname.replace("-", "__neg__")
            download_file(f["URL"], fname, 1024)

    def get_meta(self, **kwargs):
        """
        Perform a mostly arbitrary meta_data query and dump to std out
        :param kwargs:
        :return:
        """
        query = kwargs
        # filters
        query = query_params(query, "productid", None, short_hand="pid")
        query = query_params(query, "query", "product")
        query = query_params(query, "results", "m")
        query = query_params(query, "output", "j")
        return query_ode(self.ode_url, query=query)

    def get_meta_by_key(self, key, **kwargs):
        res = self.get_meta(**kwargs)
        return res[key]

    def get_ctx_meta(self, pid):
        productid = "{}*".format(pid)

        query = {
            "target": "mars",
            "query": "product",
            "results": "m",
            "output": "j",
            "pt": "EDR",
            "iid": "CTX",
            "ihid": "MRO",
            "productid": productid,
        }

        return query_ode(self.ode_url, query=query)

    def get_ctx_meta_by_key(self, pid, key):
        res = self.get_ctx_meta(pid)
        return res[key]

    def get_hirise_meta(self, pid):
        productid = "{}*".format(pid)

        query = {
            "target": "mars",
            "query": "product",
            "results": "m",
            "output": "j",
            "pt": "RDRV11",
            "iid": "HiRISE",
            "ihid": "MRO",
            "productid": productid,
        }

        return query_ode(self.ode_url, query=query)

    def get_hirise_meta_by_key(self, pid, key):
        res = self.get_hirise_meta(pid)
        return res[key]


def url_https(url):
    return url.replace("http://", "https://")


def main():
    fire.Fire(ODE)


if __name__ == "__main__":
    main()
