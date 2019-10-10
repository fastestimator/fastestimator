# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
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
import argparse
import sys

from fastestimator.cli.cli_util import load_and_caricature, load_and_saliency, load_and_umap, parse_log_dir, \
    SaveAction, parse_cli_to_dictionary, load_and_gradcam


def logs(args, unknown):
    if len(unknown) > 0:
        print("error: unrecognized arguments: ", str.join(", ", unknown))
        sys.exit(-1)
    parse_log_dir(args['log_dir'],
                  args['extension'],
                  args['recursive'],
                  args['smooth'],
                  args['save'],
                  args['save_dir'],
                  args['ignore'],
                  args['share_legend'],
                  args['pretty_names'])


def configure_log_parser(subparsers):
    parser = subparsers.add_parser('logs',
                                   description='Generates comparison graphs amongst one or more log files',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   allow_abbrev=False)
    parser.add_argument('log_dir',
                        metavar='<Log Dir>',
                        type=str,
                        help="The path to a folder containing one or more log files")
    parser.add_argument('--extension',
                        metavar='E',
                        type=str,
                        help="The file type / extension of your logs",
                        default=".txt")
    parser.add_argument('--recursive', action='store_true', help="Recursively search sub-directories for log files")
    parser.add_argument('--ignore',
                        metavar='I',
                        type=str,
                        nargs='+',
                        help="The names of metrics to ignore though they may be present in the log files")
    parser.add_argument('--smooth',
                        metavar='<float>',
                        type=float,
                        help="The amount of gaussian smoothing to apply (zero for no smoothing)",
                        default=1)
    parser.add_argument('--pretty_names', help="Clean up the metric names for display", action='store_true')

    legend_group = parser.add_argument_group('legend arguments')
    legend_x_group = legend_group.add_mutually_exclusive_group(required=False)
    legend_x_group.add_argument('--common_legend',
                                dest='share_legend',
                                help="Generate one legend total",
                                action='store_true',
                                default=True)
    legend_x_group.add_argument('--split_legend',
                                dest='share_legend',
                                help="Generate one legend per graph",
                                action='store_false',
                                default=False)

    save_group = parser.add_argument_group('output arguments')
    save_x_group = save_group.add_mutually_exclusive_group(required=False)
    save_x_group.add_argument(
        '--save',
        nargs='?',
        metavar='<Save Dir>',
        dest='save',
        action=SaveAction,
        default=False,
        help="Save the output image. May be accompanied by a directory into \
                                              which the file is saved. If no output directory is specified, the log \
                                              directory will be used")
    save_x_group.add_argument('--display',
                              dest='save',
                              action='store_false',
                              help="Render the image to the UI (rather than saving it)",
                              default=True)
    save_x_group.set_defaults(save_dir=None)
    parser.set_defaults(func=logs)


def saliency(args, unknown):
    if len(unknown) > 0:
        print("error: unrecognized arguments: ", str.join(", ", unknown))
        sys.exit(-1)
    load_and_saliency(args['model'],
                      args['inputs'],
                      baseline=args['baseline'],
                      dictionary_path=args['dictionary'],
                      strip_alpha=args['strip_alpha'],
                      smooth_factor=args['smooth'],
                      save=args['save'],
                      save_dir=args['save_dir'])


def configure_saliency_parser(subparsers):
    parser = subparsers.add_parser('saliency',
                                   description='Generates saliency maps for a model on given input(s)',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   allow_abbrev=False)
    parser.add_argument('model', metavar='<Model Path>', type=str, help="The path to a saved model file")
    parser.add_argument(
        'inputs',
        metavar='Input',
        type=str,
        nargs='+',
        help="The paths to one or more inputs to visualize (or the path to a folder which will be \
                        recursively traversed to find image files)")
    parser.add_argument('--dictionary',
                        metavar='<Dictionary Path>',
                        type=str,
                        help="The path to a {'class_id':'class_name'} json file",
                        default=None)
    parser.add_argument('--smooth',
                        metavar='N',
                        type=int,
                        default=7,
                        help="The number of samples to use when generating smoothed saliency masks")
    parser.add_argument(
        '--baseline',
        metavar='B',
        type=str,
        default='-1',
        help="The value to use as a baseline for integrated gradient calculations. Can be either a \
                        number or the path to an image. What would a 'blank' input look like")
    parser.add_argument('--strip-alpha', action='store_true', help="True if you want to convert RGBA images to RGB")
    save_group = parser.add_argument_group('output arguments')
    save_x_group = save_group.add_mutually_exclusive_group(required=False)
    save_x_group.add_argument(
        '--save',
        nargs='?',
        metavar='<Save Dir>',
        help="Save the output image. May be accompanied by a directory into which the \
                              file is saved. If no output directory is specified, the model directory will be used",
        dest='save',
        action=SaveAction,
        default=False)
    save_x_group.add_argument('--display',
                              dest='save',
                              action='store_false',
                              help="Render the image to the UI (rather than saving it)",
                              default=True)
    save_x_group.set_defaults(save_dir=None)
    parser.set_defaults(func=saliency)


def umap(args, unknown):
    hyperparameters = parse_cli_to_dictionary(unknown)
    load_and_umap(args['model'],
                  args['input_dir'],
                  print_layers=args['print_layers'],
                  strip_alpha=args['strip_alpha'],
                  layers=args['layers'],
                  batch=args['batch'],
                  use_cache=args['cache'],
                  cache_dir=args['cache_dir'],
                  dictionary_path=args['dictionary'],
                  legend_mode=args['legend'],
                  save=args['save'],
                  save_dir=args['save_dir'],
                  umap_parameters=hyperparameters,
                  input_extension=args['extension'])


def configure_umap_parser(subparsers):
    parser = subparsers.add_parser('umap',
                                   description='Plots umaps for model layers over given inputs',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   allow_abbrev=False)
    parser.add_argument('model', metavar='<Model Path>', type=str, help="The path to a saved model file")
    parser.add_argument('input_dir',
                        metavar='Input',
                        type=str,
                        help="The path to a folder of folders containing inputs sorted by class")
    parser.add_argument(
        '--print-layers',
        action='store_true',
        help="True if you only want to print a model's layers to the console. Useful for inspecting a \
                            model to choose which layer indices to visualize.")
    parser.add_argument(
        '--layers',
        metavar='IDX',
        type=int,
        nargs='+',
        help='The indices of layers you want to visualize. If not provided then all layers will be \
                                      analyzed, but that is likely to run out of memory on moderately sized models.')
    parser.add_argument('--strip-alpha', action='store_true', help="True if you want to convert RGBA images to RGB")
    parser.add_argument('--extension',
                        metavar='E',
                        type=str,
                        help="The file type / extension of your input files (default matches all files)",
                        default=None)
    parser.add_argument('--batch',
                        type=int,
                        default=50,
                        help="How many images to process at a time (set based on available RAM)")
    parser.add_argument('--dictionary',
                        metavar='<Dictionary Path>',
                        type=str,
                        help="The path to a {'class_id':'class_name'} json file",
                        default=None)
    parser.add_argument('--legend',
                        choices=['on', 'off', 'shared'],
                        default='shared',
                        help='Whether to display legends or not.')
    save_group = parser.add_argument_group('output arguments')
    save_group.add_argument(
        '--cache',
        metavar='<Cache Dir>',
        action=SaveAction,
        default=False,
        help="Enable caching of intermediate results. Highly recommended to reduce RAM \
                            requirements. If no cache directory is specified, a sibling of the input directory will be \
                            used")
    save_group.set_defaults(cache_dir=None)
    save_x_group = save_group.add_mutually_exclusive_group(required=False)
    save_x_group.add_argument(
        '--save',
        nargs='?',
        metavar='<Save Dir>',
        dest='save',
        action=SaveAction,
        default=False,
        help="Save the output image. May be accompanied by a directory into \
                                                      which the file is saved. If no output directory is specified, \
                                                      the model directory will be used")
    save_x_group.add_argument('--display',
                              dest='save',
                              action='store_false',
                              help="Render the image to the UI (rather than saving it)",
                              default=True)
    save_x_group.set_defaults(save_dir=None)
    parser.add_argument_group(
        'umap arguments',
        'Arguments from the umap module may all be passed through. Examples \
    include --n_neighbors <int>, --metric <str>, --n_epochs <int>, --min_dist <float>, --learning_rate <float>, etc...')
    parser.set_defaults(func=umap)


def caricature(args, unknown):
    if len(unknown) > 0:
        print("error: unrecognized arguments: ", str.join(", ", unknown))
        sys.exit(-1)
    load_and_caricature(args['model'],
                        args['inputs'],
                        dictionary_path=args['dictionary'],
                        save=args['save'],
                        save_dir=args['save_dir'],
                        strip_alpha=args['strip_alpha'],
                        layer_ids=args['layers'],
                        print_layers=args['print_layers'],
                        n_steps=args['steps'],
                        learning_rate=args['learning_rate'],
                        blur=args['blur'],
                        cossim_pow=args['cossim'],
                        sd=args['stddev'],
                        fft=not args['pixel_based'],
                        decorrelate=not args['true_color'],
                        sigmoid=not args['hard_clip'])


def configure_caricature_parser(subparsers):
    parser = subparsers.add_parser('caricature',
                                   description='Plots caricatures for model layers over given inputs',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   allow_abbrev=False)
    parser.add_argument('model', metavar='<Model Path>', type=str, help="The path to a saved model file")
    parser.add_argument(
        'inputs',
        metavar='Input',
        type=str,
        nargs='+',
        help="The paths to one or more inputs to visualize (or the path to a folder which will be \
                            recursively traversed to find image files)")
    parser.add_argument(
        '--print-layers',
        action='store_true',
        help="True if you only want to print a model's layers to the console. Useful for inspecting a \
                            model to choose which layer indices to visualize.")
    parser.add_argument(
        '--layers',
        metavar='IDX',
        type=int,
        nargs='+',
        help='The indices of layers you want to visualize. If not provided then all layers will be \
                                      analyzed, but that is likely to run out of memory on moderately sized models.')
    parser.add_argument('--strip-alpha', action='store_true', help="True if you want to convert RGBA images to RGB")
    parser.add_argument('--dictionary',
                        metavar='<Dictionary Path>',
                        type=str,
                        help="The path to a {'class_id':'class_name'} json file",
                        default=None)
    parser.add_argument('--steps', metavar='N', type=int, help="How many optimization iterations to run", default=250)
    parser.add_argument('--learning-rate',
                        metavar='LR',
                        type=float,
                        help="Initial learning rate for the optimizer",
                        default=0.05)
    parser.add_argument('--blur',
                        metavar='B',
                        type=float,
                        help="Strength of blurring used on image during processing (reduces high-frequency artifacts)",
                        default=1.0)
    parser.add_argument(
        '--cossim',
        metavar='B',
        type=float,
        help="Strength of cosine similarity coefficient in loss function (higher value increases \
                        similarity)",
        default=0.5)
    parser.add_argument('--stddev',
                        type=float,
                        help="The standard deviation of the noise used to generate the seed image",
                        default=0.01)
    parser.add_argument(
        '--pixel-based',
        action='store_true',
        help="Use pixel-based optimization rather than fft based. Individual steps will be faster but \
                        convergence will be slower")
    parser.add_argument('--true-color',
                        action='store_true',
                        help="Disable color decorrelation. You may want this if using non-imagenet style images")
    parser.add_argument('--hard-clip',
                        action='store_true',
                        help="Use hard clipping instead of sigmoids to constrain pixel values")
    save_group = parser.add_argument_group('output arguments')
    save_x_group = save_group.add_mutually_exclusive_group(required=False)
    save_x_group.add_argument(
        '--save',
        nargs='?',
        metavar='<Save Dir>',
        help="Save the output image. May be accompanied by a directory into which the \
                              file is saved. If no output directory is specified, the model directory will be used",
        dest='save',
        action=SaveAction,
        default=False)
    save_x_group.add_argument('--display',
                              dest='save',
                              action='store_false',
                              help="Render the image to the UI (rather than saving it)",
                              default=True)
    save_x_group.set_defaults(save_dir=None)
    parser.set_defaults(func=caricature)


def gradcam(args, unknown):
    if len(unknown) > 0:
        print("error: unrecognized arguments: ", str.join(", ", unknown))
        sys.exit(-1)
    load_and_gradcam(args['model'],
                     args['inputs'],
                     layer_id=args['layer'],
                     dictionary_path=args['dictionary'],
                     strip_alpha=args['strip_alpha'],
                     save=args['save'],
                     save_dir=args['save_dir'])


def configure_gradcam_parser(subparsers):
    parser = subparsers.add_parser('gradcam',
                                   description='Generates gradcam maps for a model on given input(s)',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   allow_abbrev=False)
    parser.add_argument('model', metavar='<Model Path>', type=str, help="The path to a saved model file")
    parser.add_argument(
        'inputs',
        metavar='Input',
        type=str,
        nargs='+',
        help="The paths to one or more inputs to visualize (or the path to a folder which will be \
                        recursively traversed to find image files)")
    parser.add_argument(
        '--layer',
        metavar='IDX',
        type=int,
        nargs=1,
        help='The index of (convolutional) layer you want to visualize. If not provided then the last conv layer will \
                be used',
        default=None)
    parser.add_argument('--dictionary',
                        metavar='<Dictionary Path>',
                        type=str,
                        help="The path to a {'class_id':'class_name'} json file",
                        default=None)
    parser.add_argument('--strip-alpha', action='store_true', help="True if you want to convert RGBA images to RGB")
    save_group = parser.add_argument_group('output arguments')
    save_x_group = save_group.add_mutually_exclusive_group(required=False)
    save_x_group.add_argument(
        '--save',
        nargs='?',
        metavar='<Save Dir>',
        help="Save the output image. May be accompanied by a directory into which the \
                              file is saved. If no output directory is specified, the model directory will be used",
        dest='save',
        action=SaveAction,
        default=False)
    save_x_group.add_argument('--display',
                              dest='save',
                              action='store_false',
                              help="Render the image to the UI (rather than saving it)",
                              default=True)
    save_x_group.set_defaults(save_dir=None)
    parser.set_defaults(func=gradcam)


def configure_visualization_parser(subparsers):
    visualization_parser = subparsers.add_parser('visualize',
                                                 description='Generates various types of visualiztaions',
                                                 allow_abbrev=False)
    visualization_subparsers = visualization_parser.add_subparsers()
    # In python 3.7 the following 2 lines could be put into the .add_subparsers() call
    visualization_subparsers.required = True
    visualization_subparsers.dest = 'mode'

    configure_log_parser(visualization_subparsers)
    configure_saliency_parser(visualization_subparsers)
    configure_umap_parser(visualization_subparsers)
    configure_caricature_parser(visualization_subparsers)
    configure_gradcam_parser(visualization_subparsers)
