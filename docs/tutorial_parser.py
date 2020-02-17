""" FastEstimator Docstring parser """
import glob
import inspect
import json
import os
import pydoc
import re
import subprocess
import sys
import tempfile

fe_dir = 'fastestimator'
tutorial_md_dir = 'tutorials_md'
tmp_output = '/var/lib/jenkins/workspace/tmp_output'
tmp_markdown = os.path.join(tmp_output, 'tutorial')
struct_json_path = os.path.join(tmp_markdown, 'structure.json')

re_sidebar_title = '[^A-Za-z0-9:!,$%. ]+'
re_route_title = '[^A-Za-z0-9 ]+'

fe_path = os.path.join(os.getcwd(), fe_dir)

tutorial_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../tutorial/*.ipynb')

subprocess.run(['jupyter', 'nbconvert', '--to', 'markdown', tutorial_path, '--output-dir', tmp_markdown])


def replaceImagePath(mdfile):
    """This function takes markdown file path and append the prefix path to the image path in the file. It allows
    angular to find images in the server.

    Args:
        mdfile: markdown file
    """
    mdcontent = open(mdfile).readlines()
    png_tag = '![png]('
    html_img_tag = '<img src="'
    path_prefix = 'assets/tutorial'
    mdfile_updated = []
    for line in mdcontent:
        idx1, idx2 = map(line.find, [png_tag, html_img_tag])
        if idx1 != -1:
            line = png_tag + os.path.join(path_prefix, line[idx1 + len(png_tag):])
            mdfile_updated.append(line)
        elif idx2 != -1:
            line = html_img_tag + os.path.join(path_prefix, line[idx2 + len(html_img_tag):])
            mdfile_updated.append(line)
        else:
            mdfile_updated.append(line)
    with open(mdfile, 'w') as f:
        f.write("".join(mdfile_updated))


tutorial_list = [file for file in glob.glob(os.path.join(tmp_markdown, "*.md"))]

for f in tutorial_list:
    replaceImagePath(f)

headers = ['#']
subheaders = ['##', '###']
#structure_json = {}
#print(tutorial_list)
nav_list = []
for f in sorted(tutorial_list):
    out_headers = []
    mdfile = open(f).readlines()
    flag = True
    structure_json = {}
    sidebar_titles = []

    for sentence in mdfile:
        sentence = sentence.strip()
        sentence_tokens = sentence.split(' ')
        #print(sentence_tokens)
        #if sentence_tokens[0] in headers:
        sidebar_val_dict = {}
        if flag and sentence_tokens[0] in headers:
            structure_json['name'] = os.path.basename(f)
            structure_json['displayName'] = re.sub(re_sidebar_title, '', sentence)
            flag = False
        elif sentence_tokens[0] in subheaders:
            title = re.sub(re_sidebar_title, '', sentence)
            route_title = re.sub(re_route_title, '', sentence)
            sidebar_val_dict['id'] = route_title.lower().strip().replace(' ', '-')
            sidebar_val_dict['displayName'] = title.strip()
            sidebar_titles.append(sidebar_val_dict)
    structure_json['toc'] = sidebar_titles
    nav_list.append(structure_json)

#write to json file
with open(struct_json_path, 'w') as f:
    f.write(json.dumps(nav_list))
