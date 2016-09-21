import os
import pprint

from datasets.meta import ImageMeta


class RoiDb(object):
  def __init__(self, image_set, year, devkit_path='data/VOCdevkit'):
    self._classes = ('__background__',  # always index 0
                     'aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor')
    self._image_set = image_set
    self._year = year
    self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
    self._devkit_path = devkit_path

  @property
  def classes(self):
    return self._classes

  @property
  def num_classes(self):
    return len(self.classes)

  def _load_image_ids(self):
    voc_path = os.path.join(self._devkit_path, 'VOC%s' % self._year, 'ImageSets/Main', self._image_set)

    assert os.path.exists(voc_path), "Dataset not existed."

    with open(voc_path) as file:
      return [line.strip() for line in file.readlines()]

  def _meta_path(self, image_id):
    annotation_dir = os.path.join(self._devkit_path, 'VOC%s' % self._year, 'Annotations')
    return os.path.join(annotation_dir, '%s.xml' % image_id)

  @property
  def rois(self):
    image_ids = self._load_image_ids()
    return [self._generate_meta(image_id) for image_id in image_ids]

  def _generate_meta(self, image_id):
    meta = ImageMeta(image_id, self._image_path(image_id), self._meta_path(image_id))
    for obj in meta.objects:
      obj['class_index'] = self._class_to_ind[obj['class']]
    return meta

  def _image_path(self, image_id):
    image_dir = os.path.join(self._devkit_path, 'VOC%s' % self._year, 'JPEGImages')
    return os.path.join(image_dir, '%s.jpg' % image_id)


if __name__ == '__main__':
  print(os.path.abspath(os.path.curdir))
  roidb = RoiDb('val.txt', 2007)
  print(len(roidb.rois))
  pprint.pprint(roidb.rois[0].image_path)
  pprint.pprint(roidb.rois[0].objects)
  pprint.pprint(roidb.rois[0].shape)
