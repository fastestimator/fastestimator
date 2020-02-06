""" FastEstimator Apphub parser """
import glob
import inspect
import json
import os
import pydoc
import re
import subprocess
import sys
import tempfile

re_sidebar_title = '[^A-Za-z0-9:!,$%.() ]+'
re_route_title = '[^A-Za-z0-9 ]+'
re_url = '\[notebook\]\((.*)\)'

apphub_md_dir = 'apphub_md'
tmp_markdown = '../../tmp_output'
#tmp_markdown = os.path.join(tempfile.gettempdir(), apphub_md_dir)
struct_json_path = os.path.join(tmp_markdown, 'structure_apphub.json')
apphub_toc_path = os.path.join(tmp_markdown, 'apphub_toc.md')
apphub_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../apphub/')


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def replaceImagePath(mdfile, d):
    mdcontent = open(mdfile).readlines()
    png_tag = '![png]('
    html_img_tag = '<img src="'
    path_prefix = os.path.join('assets/example', d)
    mdfile_updated = []
    for line in mdcontent:
        idx1, idx2 = map(line.find, [png_tag, html_img_tag])
        if idx1 != -1 and line.split(os.path.sep)[0] != 'assets':
            line = png_tag + os.path.join(path_prefix, line[idx1 + len(png_tag):])
            mdfile_updated.append(line)
        elif idx2 != -1 and line.split(os.path.sep)[0] != 'assets':
            line = html_img_tag + os.path.join(path_prefix, line[idx2 + len(html_img_tag):])
            mdfile_updated.append(line)
        else:
            mdfile_updated.append(line)
    with open(mdfile, 'w') as f:
        f.write("".join(mdfile_updated))
    return mdfile


def extractTitle(md_path, fname):
    headers = ['#']
    f = os.path.join(md_path, fname + '.md')
    mdfile = open(f).readlines()
    for sentence in mdfile:
        sentence = sentence.strip()
        sentence_tokens = sentence.split(' ')
        if sentence_tokens[0] in headers:
            title = re.sub(re_sidebar_title, '', sentence)
            return title


def extractReadMe(apphub_path):
    readmefile = os.path.join(apphub_path, 'README.md')
    content = open(readmefile).readlines()
    toc = []
    startidx = 0
    endidx = len(content) - 1
    for i, line in enumerate(content):
        idx = line.find('Table of Contents:')
        if idx != -1:
            startidx = i
        elif line.split(' ')[0] == '##' and startidx != 0:
            endidx = i
            break
    toc = content[startidx:endidx]

    #write table of content to the file
    with open(apphub_toc_path, 'w') as f:
        f.write("".join(toc))
    ex_list = []
    title_dict = {}
    child, files_list = [], []
    flag = False
    for line in toc[1:]:
        line_tokens = line.split(' ')
        if line_tokens[0] in ['###', '####']:
            if flag:
                title_dict['children'] = child
                ex_list.append(title_dict)
                child = []
                title_dict = {}
            title = re.sub(re_sidebar_title, '', line).strip()
            title_dict['title'] = title
        else:
            flag = True
            l = line.split(':')[0]
            name = re.sub(re_sidebar_title, '', l)
            url = re.findall(re_url, line)
            if name != '' and url:
                fname = os.path.basename(url[0]).split('.')[0]
                child_dict = {}
                child_dict[fname] = {'title': title, 'name': name.strip()}
                files_list.append(child_dict)
                child.append(child_dict)
    return files_list


json_struct = []
exclude_prefixes = ['_', '.']
for subdirs, dirs, files in os.walk(apphub_path, topdown=True):
    dirs[:] = [d for d in dirs if not d[0] in exclude_prefixes]
    for f in files:
        fname, ext = os.path.splitext(os.path.basename(f))
        if ext == '.ipynb' and f[0] != '.':
            example_type = splitall(os.path.relpath(subdirs, apphub_path))[0]
            save_dir = os.path.join(tmp_markdown, example_type)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            subprocess.run(
                ['jupyter', 'nbconvert', '--to', 'markdown', os.path.join(subdirs, f), '--output-dir', save_dir])


def getNameTitle(namedict, fname):
    for obj in namedict:
        if fname in obj.keys():
            title = obj[fname]['title']
            name = obj[fname]['name']
            return title, name


def imagePathModify():
    for subdirs, dirs, files in os.walk(tmp_markdown, topdown=True):
        for f in files:
            if f.endswith('.md'):
                d = subdirs.split(os.path.sep)[-1]
                replaceImagePath(os.path.join(subdirs, f), d)


def create_json(path):
    json_dict = {}
    json_struct = []
    json_struct.append({'displayName': 'Overview', 'name': 'overview.md'})
    exclude_prefixes = ['_', '.']
    namedict = extractReadMe(apphub_path)
    for d in os.listdir(path):
        child_list = []
        parent_json_obj = {}
        if os.path.isdir(os.path.join(path, d)):
            files = [
                f for f in os.listdir(os.path.join(path, d)) if os.path.isfile(os.path.join(*[tmp_markdown, d, f]))
            ]
            for f in files:
                file_json_obj = {}
                fname, ext = os.path.splitext(os.path.basename(f))
                title, name = getNameTitle(namedict, fname)
                if ext == '.md' and f[0] != '.':
                    file_json_obj['name'] = os.path.join(d, fname + ext)
                    file_json_obj['displayName'] = name
                    child_list.append(file_json_obj)
            parent_json_obj['displayName'] = title
            parent_json_obj['name'] = d
            parent_json_obj['children'] = child_list
            json_struct.append(parent_json_obj)
    return json_struct


imagePathModify()
#write to json file
with open(struct_json_path, 'w') as f:
    f.write(json.dumps(create_json(tmp_markdown)))
