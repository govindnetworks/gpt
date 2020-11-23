#----------------------------------------------#
# Radiation Detection and Imaging technologies #
#                                              #
#----------------------------------------------#

#  core library imports
import sys
import random as rand

# special library imports
import numpy as np
from scipy import spatial



# read in pad file
with open('../In/3d.kicad_pcb') as f:
    text = f.read()


class PcbObject:
    def __repr__(self):
        """

        :return: string
        """
        return "{}".format(self.__dict__)

def skip_content(s, begin):
    """
    skip the inner pattern of '(' * ')'
    :param s:
    :param begin:
    :return:
    """
    while True:
        l = s.find('(', begin)
        r = s.find(')', begin)
        # l == -1 means end of close bracket
        if r < l or l == -1:
            return r + 1
        #change begin to next content
        begin = skip_content(s, l + 1)


def build_tree(s, begin, depth):
    """
     build tree of net and via objects
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
        #print(obj.name)
        if depth != 1 or obj.name == 'via' or obj.name == 'net':
            obj.content, begin = build_tree(s, space + 1, depth + 1)
            pcbobj.append(obj)
        else:
            begin = skip_content(s, space + 1)


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
axis_map = {'a': 1, 'b': 2, 'c': 0} # mapping axis with a20, b21
axis_flip = {'a': 1, 'b': -1, 'c': 1}
gnd_net = 'GND'  #end of net
#print(kicad_pcb.content[2])
for obj in kicad_pcb.content:
    #print("obj is {}".format(obj))
    if obj.name == 'net':
        #{'name': 'net', 'content': '61 b0'}
        #{'name': 'net', 'content': '60 a0'}
        split = obj.content.split()
        net_map[int(split[0])] = split[1]
    elif obj.name == 'via':
        net = ''
        at = (0, 0)
        for obj1 in obj.content:
            if obj1.name == 'at':
                #{'name': 'via', 'content': [{'name': 'at', 'content': '41.877219 22.386874'},
                # {'name': 'size', 'content': '0.25'}, {'name': 'drill', 'content': '0.15'},
                # {'name': 'layers', 'content': 'F.Cu In7.Cu'}, {'name': 'net', 'content': '15'}]}
                split = obj1.content.split()
                at = (float(split[0]), float(split[1]))
            elif obj1.name == 'net':
                net = net_map[int(obj1.content.split()[0])]
                print(net)
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

# print(pad_a)
# print("*"*30)
# print(pad_x)
# print("*"*30)
# print(pad_y)
print("*"*30)

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

out = np.transpose([pad_x, pad_y])
np.savetxt("kicad_matrix.csv", out, delimiter=",")

# define nearest channel sample binning using native KDTree search
pad_tree = spatial.cKDTree(np.transpose([pad_x, pad_y]))
print(pad_tree)

#Not used below methods
def chan_counts(xy, split=True, randomize=True):
    if randomize:
        np.random.shuffle(xy)

    # split the samples evenly between the anode and cathode
    if split:
        xy = xy[:len(xy) // 2]

    _, pads = pad_tree.query(xy)
    pads, counts = np.unique(pads, return_counts=True)
    try:
        pads, counts = zip(*filter(lambda x: x[0] < len(pad_c), zip(pads, counts)))
        return zip(pad_c[list(pads)], counts)
    except ValueError:
        return []