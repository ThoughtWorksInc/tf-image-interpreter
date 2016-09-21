import xml.etree.ElementTree as ET


def _parse_object(elem):
  bbox_elem = elem.find('bndbox')
  return {
    'class': elem.find('name').text.lower().strip(),
    'bbox': {
      float(bbox_elem.find('xmin').text) - 1,
      float(bbox_elem.find('ymin').text) - 1,
      float(bbox_elem.find('xmax').text) - 1,
      float(bbox_elem.find('ymax').text) - 1,
    }
  }


def _parse_shape(elem):
  return [elem.find('width'),
          elem.find('height'),
          elem.find('depth')]


class ImageMeta(object):
  def __init__(self, index, image_path, meta_path):
    self._index = index
    self._meta_path = meta_path
    self._image_path = image_path
    self._parse_meta(meta_path)

  def _parse_meta(self, meta_path):
    tree = ET.parse(meta_path)
    self._shape = _parse_shape(tree.find('size'))
    obj_elems = tree.findall('object')
    self._objs = [_parse_object(elem) for elem in obj_elems]

  @property
  def image_path(self):
      return self._image_path

  @property
  def shape(self):
      return self._shape

  @property
  def objects(self):
      return self._objs
