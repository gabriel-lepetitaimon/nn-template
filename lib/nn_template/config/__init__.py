from . import cfg_object
from . import cfg_parser
from . import cfg_dict

from .cfg_dict import CfgDict, UNDEFINED
from .cfg_parser import CfgParser, CfgFile, register_obj


class Cfg:
    Dict = cfg_dict.CfgDict
    Collection = cfg_dict.CfgCollection
    List = cfg_dict.CfgList
    UNDEFINED = cfg_dict.UNDEFINED

    Obj = cfg_object.CfgObj
    Attr = cfg_object.CfgAttr
    InvalidAttr = cfg_object.InvalidAttr

    int = cfg_object.IntAttr
    float = cfg_object.FloatAttr
    str = cfg_object.StrAttr
    range = cfg_object.RangeAttr
    oneOf = cfg_object.OneOfAttr
    bool = cfg_object.BoolAttr
    strMap = cfg_object.StrMapAttr
    list = cfg_object.ListAttr
    strList = cfg_object.StrListAttr
    obj = cfg_object.ObjAttr
    any = cfg_object.AnyAttr
    collection = cfg_object.CollectionAttr
    obj_list = cfg_object.ObjListAttr
    multi_type_collection = cfg_object.MultiTypeCollectionAttr
    shape = cfg_object.ShapeAttr
    ref = cfg_object.RefAttr

    register_obj = cfg_parser.register_obj
    Parser = cfg_parser.CfgParser
