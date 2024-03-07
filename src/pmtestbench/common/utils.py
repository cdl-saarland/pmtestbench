import os
import re

def popcount(n):
    """ Return the number of 1s in the binary representation of the number n.
    """
    return bin(n).count("1")


def increase_filename(fn):
    """ Create a reasonable successor filename for the given filename. """
    name, ext = os.path.splitext(fn)
    m = re.fullmatch(r"(.*_)(\d+)", name)
    if m is None:
        return name + "_01" + ext
    else:
        name_match = m[1]
        num_match = m[2]
        num = int(num_match)
        return name_match + ("{:0" + str(len(num_match)) + "d}").format(num + 1) + ext


