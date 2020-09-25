# Copyright 2020 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import pickle
import unittest
import urllib.request
import warnings


def ping(host):
    try:
        agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'
        req = urllib.request.Request(host)
        req.add_header('User-Agent', agent)
        code = urllib.request.urlopen(req).getcode()
        return code == 200
    except urllib.request.HTTPError as e:
        return False
    return False


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        urls_dict_path = os.path.abspath(os.path.join(__file__, "..", "resources", "dataset_urls.pickle"))
        with open(urls_dict_path, 'rb') as dataset_url_dict:
            cls.data_urls = pickle.load(dataset_url_dict)

    def test_dataset_urls(self):
        for key, value in self.data_urls.items():
            if isinstance(value, list):
                for url in value:
                    with self.subTest('{}{} url'.format(key, url)):
                        self.assertTrue(ping(url))
            else:
                with self.subTest('Check if {} url reachable'.format(key)):
                    self.assertTrue(ping(value))
