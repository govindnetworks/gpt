#----------------------------------------------#
# RDI                                          #
#                                              #
#----------------------------------------------#

#  core library imports
import logging
import sys
import os

# special library imports
import numpy as np
import random as rand
from scipy import spatial
import matplotlib.pyplot as plt
import memory_profiler
from memory_profiler import profile

#custom library import
import nngpt

class PcbObject:
    pass

def read_pad_file():
    """
    Read the file and return list of string
    :param filename:
    :return:
    """
    # read in pad file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    in_dir = os.path.join(dir_path, 'In')
    out_dir = os.path.join(dir_path, 'Out')
    print(in_dir, out_dir,dir_path)
    #get all the files in In folder
    files_path = [os.path.join(in_dir,x) for x in os.listdir(in_dir) if x.endswith(".kicad_pcb")]
    print(files_path)
    for count, name in enumerate(files_path):
        print("{} :Filename {}".format(count, name))
    sel_opt = int(input("Enter number to process kicad file:"))
    file_name = files_path[sel_opt]
    #Print selected files
    print("The selected file {}".format(file_name))
    with open(file_name, "r") as fp:
        text = fp.read()
    _,out_name = os.path.split(file_name)
    out_file = os.path.join(out_dir,out_name)
    #move into outfolder
    os.rename(file_name,out_file)
    return text

def skip_content(s, begin):
    """

    :param s:
    :param begin:
    :return:
    """
    #skip the contents ( * )
    while True:
        l = s.find('(', begin)
        r = s.find(')', begin)
        if r < l or l == -1:
            return r + 1
        begin = skip_content(s, l + 1)

def build_tree(s, begin, depth):
    """
     Build data structure data from unstructre text file
    :param s:
    :param begin:
    :param depth:
    :return:
    """
    pcbobj = []
    while True:
        l = s.find('(', begin)
        r = s.find(')', begin)
        if r < l or l == -1:
            if len(pcbobj) == 0:
                return s[begin:r], r + 1
            else:
                return pcbobj, r + 1
        begin = l + 1
        space = s.find(' ', begin)
        obj = PcbObject()
        obj.name = s[begin:space].strip()
        if depth != 1 or obj.name == 'via' or obj.name == 'net':
            obj.content, begin = build_tree(s, space + 1, depth + 1)
            pcbobj.append(obj)
        else:
            begin = skip_content(s, space + 1)

def build_kd_tree(text):
    """
    build kd tree using scipy module
    :param text:
    :return:
    """
    pcb_tree, _ = build_tree(text, 0, 0)
    if len(pcb_tree) != 1:
        raise Exception('bad pcb tree structure with length', len(pcb_tree))
    if pcb_tree[0].name != 'kicad_pcb':
        raise Exception('not a kicad pcb file')
    kicad_pcb = pcb_tree[0]

    # build arrays mapping pads to channel numbers
    min_c = [sys.maxsize, sys.maxsize, sys.maxsize]
    pad_a = []
    pad_ac = []
    pad_x = []
    pad_y = []
    net_map = {}
    axis_map = {'a': 1, 'b': 2, 'c': 0}
    axis_flip = {'a': 1, 'b': -1, 'c': 1}
    gnd_net = 'GND'

    for obj in kicad_pcb.content:
        #{'name': 'net', 'content': '61 b0'}
        #{'name': 'net', 'content': '60 a0'}
        if obj.name == 'net':
            split = obj.content.split()
            net_map[int(split[0])] = split[1]
        elif obj.name == 'via':
            net = ''
            at = (0, 0)
            #{'name': 'via', 'content': [{'name': 'at', 'content': '41.877219 22.386874'},
            # {'name': 'size', 'content': '0.25'}, {'name': 'drill', 'content': '0.15'},
            # {'name': 'layers', 'content': 'F.Cu In7.Cu'}, {'name': 'net', 'content': '15'}]}
            for obj1 in obj.content:
                if obj1.name == 'at':
                    split = obj1.content.split()
                    at = (float(split[0]), float(split[1]))
                elif obj1.name == 'net':
                    net = net_map[int(obj1.content.split()[0])]
            if net[0] in axis_map:
                try:
                    axis = axis_map[net[0]]
                    channel = axis_flip[net[0]] * int(net[1:])
                    if channel < min_c[axis]:
                        min_c[axis] = channel
                    pad_a.append(axis)
                    pad_ac.append(channel)
                    pad_x.append(at[0])
                    pad_y.append(at[1])
                except:
                    continue
            elif net == gnd_net:
                pad_a.append(-1)
                pad_ac.append(-1)
                pad_x.append(at[0])
                pad_y.append(at[1])

    pad_ac = np.array(pad_ac)
    n_axis_chans = len(np.unique(pad_ac[np.flatnonzero(np.array(pad_a) == 0)]))

    pad_c = [-1] * len(pad_ac)
    for i in range(0, len(pad_ac)):
        a = pad_a[i]
        ac = pad_ac[i] - min_c[a]
        if a >= 0 and ac >= 0:
            pad_c[i] = n_axis_chans * a + ac

    pad_x = np.array(pad_x)
    pad_y = np.array(pad_y)
    pad_c = np.array(pad_c)

    chans = np.unique(pad_c)
    n_chans = len(chans)
    if -1 in chans:
        n_chans -= 1

    # define nearest channel sample binning using native KDTree search
    pad_tree = spatial.cKDTree(np.transpose([pad_x, pad_y]))
    return pad_tree, n_chans,pad_c
"""
#moved into nngpt module must be removed
  def chan_counts(xy, pd_tree, pad_c, split=True, randomize=True):
    if randomize:
        np.random.shuffle(xy)

    # split the samples evenly between the anode and cathode
    if split:
        xy = xy[:len(xy) // 2]

    _, pads = pd_tree.query(xy)
    pads, counts = np.unique(pads, return_counts=True)
    try:
        pads, counts = zip(*filter(lambda x: x[0] < len(pad_c), zip(pads, counts)))
        return zip(pad_c[list(pads)], counts)
    except ValueError:
        return []
*/"""

def plot_tomo_for_samples(xy, name, n_std=4, seg_name='3-D'):
    logging.getLogger().setLevel(logging.INFO)

    xy_diff = nngpt.add_diffusion(xy, p.max_diff_sigma)
    q = p.bin_channels(xy_diff)
    img = p.tomo(q)
    mean, cov = p.get_mean_and_cov(img)
    true_img = p.bin_pixels(xy)
    true_mean, true_cov = p.get_mean_and_cov(true_img)

    print('Simulated channel input')
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot()
    ax.plot(q)
    ax.set_xlabel('readout channel')
    ax.set_yscale('log')
    fig.savefig(f'{seg_name}_{name}_channel-response.pdf', bbox_inches='tight', pad_inches=0)
    #plt.show()

    print(f'True image with calculated {n_std}-sigma confidence ellipse')
    fig = nngpt.draw_tomo(
        true_img, p.width, p.height, colorbar=True,
        mean=[None], cov=[None],
        true_mean=[[true_mean]], true_cov=[[true_cov]],
        n_std=n_std,
    )
    for ax in fig.axes:
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        break
    fig.savefig(f'{name}_truth-img.pdf', bbox_inches='tight', pad_inches=0)
    #plt.show()

    print(f'Reconstructed image with overlayed confidence ellipses')
    fig = nngpt.draw_tomo(
        img, p.width, p.height, colorbar=True,
        true_mean=[[true_mean]], true_cov=[[true_cov]],
        mean=[[mean]], cov=[[cov]],
        n_std=n_std,
    )
    for ax in fig.axes:
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        break
    fig.savefig(f'{seg_name}_{name}_reco-img.pdf', bbox_inches='tight', pad_inches=0)
    #plt.show()

    print('Differences in the images and overlayed confidence ellipses')
    fig = nngpt.draw_tomo(
        img - true_img, p.width, p.height, colorbar=True,
    )
    for ax in fig.axes:
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        break
    fig.savefig(f'{seg_name}_{name}_diff-img.pdf', bbox_inches='tight', pad_inches=0)
    #plt.show()

def draw_cov_gauss():
    """
    use stimulated data to create covarience gauss
    :return:
    """
    means = np.array([[0, 0]])
    covs = 9 * np.array([[[1, -0.5], [-0.5, 1]]])
    n_samples = [1e6]

    xy = nngpt.sample_normal(means, covs, n_samples)
    plot_tomo_for_samples(xy, 'cov-gauss')

def draw_double_gauss():
    """
    use stimulated data for double gauss ontop pcb data
    :return:
    """
    means = 20 * np.array([[1, 1], [-1, -1]])
    covs = 4 * np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    n_samples = [1e6, 1e6]

    xy = nngpt.sample_normal(means, covs, n_samples)
    plot_tomo_for_samples(xy, 'double-gauss')

def draw_circle_gauss():
    """
    draw circle ontop pcb data
    :return:
    """
    xy = nngpt.sample_circle()
    plot_tomo_for_samples(xy, 'circle')

def draw_square_gauss():
    """
    draw square on top of pcb data
    :return:
    """
    xy = nngpt.sample_square()
    plot_tomo_for_samples(xy, 'square')

if __name__ == '__main__':
    #read from file
    output = read_pad_file()
    tree_struct, n_chans,pad_c = build_kd_tree(output)
    # initialize planar nonnegative gaussian process tomography
    p = nngpt.Planar(
        n_chans,
        tree_struct,
        pad_c,
        m=100, n=100,
        sample_density=10000,
    )
    draw_cov_gauss()

    #

