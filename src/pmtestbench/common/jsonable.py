""" An interface for objects that can be serialized to and from json.
"""

from abc import ABC, abstractmethod

import datetime
import json

from iwho.configurable import pretty_print

class JSONable(ABC):

    def __init__(self):
        self.metadata = None

    def add_metadata(self):
        if self.metadata is not None:
            return
        # add some interesting metadata
        res = dict()
        res["creation_date"] = datetime.datetime.now().isoformat()
        self.metadata = res

    @abstractmethod
    def from_json_dict(self, jsondict):
        """ Initialize an empty object from a json dictionary.

        Needs to be implemented by the subclass to be compatible with to_json_dict.
        """
        pass

    @abstractmethod
    def to_json_dict(self):
        """ Return a json representation for the object.

        Needs to be implemented by the subclass to be compatible with from_json_dict.
        """
        pass

    def to_json_str(self):
        jsondict = self.to_json_dict()
        if self.metadata is not None:
            annotated_dict = { k: v for k, v in jsondict.items() }
            assert "metadata" not in annotated_dict
            annotated_dict["metadata"] = self.metadata
            return obj_to_json_str(annotated_dict)
        else:
            return obj_to_json_str(jsondict)

    def __str__(self):
        jsondict = self.to_json_dict()
        return pretty_print(jsondict)
        # return self.to_json_str()
        # return json.dumps(jsondict, indent=2, separators=(",", ": "))

    @classmethod
    def from_json(cls, infile, *args, **kwargs):
        jsondict = json.load(infile)
        res = cls(*args, **kwargs)
        res.from_json_dict(jsondict)
        return res

    @classmethod
    def from_json_str(cls, instring, *args, **kwargs):
        jsondict = json.loads(instring)
        res = cls(*args, **kwargs)
        res.from_json_dict(jsondict)
        return res

    def to_json(self, outfile):
        self.add_metadata()
        jsondict = self.to_json_dict()
        outfile.write(self.to_json_str())
        # json.dump(jsondict, outfile, indent=2, separators=(",", ": "))


indent_str = "  "

def obj_to_json_str(obj, indent=0):
    if isinstance(obj, dict):
        res = "{\n" + (indent + 1) * indent_str
        first = True
        for k, v in obj.items():
            if not first:
                res += ",\n" + (indent + 1) * indent_str
            first = False
            res += obj_to_json_str(k, indent + 1)
            res += ": "
            res += obj_to_json_str(v, indent + 1)
        res += "\n" + indent * indent_str + "}"
    elif isinstance(obj, list):
        res = "[\n" + (indent + 1) * indent_str
        first = True
        for v in obj:
            if not first:
                res += ",\n" + (indent + 1) * indent_str
            first = False
            res += obj_to_json_str(v, indent + 1)
        res += "\n" + indent * indent_str + "]"
    elif isinstance(obj, JSONable):
        res = obj_to_json_str(obj.to_json_dict())
    elif isinstance(obj, int):
        res = '{}'.format(obj)
    else:
        res = json.dumps(obj)

    squashed_res = " ".join(res.split())
    if len(squashed_res) <= 80:
        return squashed_res
    return res
