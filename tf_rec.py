import math
import os
from glob import glob
from io import BytesIO

import albumentations as albu
import cv2
import numpy as np

import tensorflow as tf


def _bytes_feature(value):
  """string / byte 型から byte_list を返す"""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(img, mask):
  """書き出し用のfeatureを作る"""
  feature = {
    'img' : _bytes_feature(img),
    'mask' : _bytes_feature(mask)
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto


def np2byte(ndarray) -> bytes:
  """
  ndarray配列からbyte列を作る
  np.tostringでは形状を記憶できないのでBytesIOで変換する
  """
  x = BytesIO()
  np.save(x, ndarray)
  return x.getvalue()


#@tf.function
def byte2np(tf_tensor):
  """byte列からndarray配列へ変換する"""
  x = BytesIO(tf_tensor.numpy())
  y = np.load(x, allow_pickle=True)

  return y

def parse_features(example):
  """tfrecordファイルからndarray配列を読み込む"""
  features = tf.io.parse_single_example(example, features={
      'img' : tf.io.FixedLenFeature([], tf.string),
      'mask' : tf.io.FixedLenFeature([], tf.string)
  })
  img = features['img']
  mask = features['mask']
  try:
    print(img.numpy())
  except AttributeError:
    print(img)
  return img, mask


def tf_parse_features(example):
  """tfrecordファイルからndarray配列を読み込む"""
  features = tf.io.parse_single_example(example, features={
      'img' : tf.io.FixedLenFeature([], tf.string),
      'mask' : tf.io.FixedLenFeature([], tf.string)
  })
  img = features['img']
  mask = features['mask']
  return img, mask


def convert_mask(mask, color_dic):
  cls = len(color_dic) + 1
  iden_arr = np.identity(len(color_dic) + 1, dtype=np.float32)
  mask2 = np.zeros((mask.shape[0], mask.shape[1], cls), dtype=np.float32)
  mask2[:,:,0] = 1.

  for i in range(1,cls):
    mask2[np.all(mask == color_dic[i], axis=2)] = iden_arr[i]

  return mask2

def func_(x, y):
  print('x__________', x, y)
  #image = tf.image.decode_jpeg(x, channels=3)
  #mask = tf.image.decode_jpeg(y, channels=3)
  try:
    x = x.numpy()
  except AttributeError:
    print('1a')
  return x, y

def main():
  size = (432,432)
  color_dic = {1:[255,255,255]}

  img_paths = [p.replace('\\', '/') for p in glob('dataset/train/img_aug/**', recursive=True) if os.path.isfile(p)]
  mask_paths = list(map(lambda x: x.replace('/img_aug/', '/mask_aug/'), img_paths))

  batch_size = 16
  splits = math.ceil(len(img_paths)/batch_size)

  empty = []

  # albumentation
  # https://qiita.com/kurilab/items/b69e1be8d0224ae139ad
  transforms = albu.OneOf([
                  albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=90),
                  albu.GaussNoise(),
                  albu.ISONoise(intensity=(0.7,0.9)),
                  albu.Downscale(),
                  albu.ElasticTransform(),
                  albu.GaussianBlur(),
                  albu.MultiplicativeNoise(multiplier=(2.0,3.0)),
                  ])

  for i in range(splits):
    tfrecord_fname = '_record_' + str(i) + '.tfrecord'
    save_path = os.path.join('dataset', tfrecord_fname)

    # tfrecordのファイルは(画像データ数 / バッチサイズ)分作成する
    with tf.io.TFRecordWriter(tfrecord_fname) as writer:
      for img_d, mask_d in zip(img_paths[i::splits], mask_paths[i::splits]):
        # 画像変形
        img = cv2.imread(img_d)
        mask = cv2.imread(mask_d)
        #augmented = transforms(image=img, mask=mask)
        #img, mask = augmented['image'], augmented['mask']
        img = cv2.resize(img, (size[0], size[1]), cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (size[0], size[1]), cv2.INTER_NEAREST)
        # byte列に変換
        img = np2byte(img)
        mask = np2byte(mask)
        #img = np2byte(np.float32(img/127.5 - 1))
        #mask = np2byte(convert_mask(mask, color_dic))
        # シリアライズして書き出し
        proto = serialize_example(img, mask)
        writer.write(proto.SerializeToString())
    if i>2 : break


def load_gen(filename=None):
  #dataset = tf.data.Dataset.from_generator()
  batch_size = 16
  dataset = tf.data.Dataset.from_tensor_slices(['_record_0.tfrecord'])\
    .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .map(tf_parse_features, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .map(func_, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(8).prefetch(1).repeat(-1)
  #print(type(dataset))
  #iters = dataset.as_numpy_iterator() 
  #x = next(iters)
  #print(type(x), len(x))

  i = 0
  for x, y in dataset:
    print(len(x), len(y))
    if i==0:
      print(byte2np(x[0]).shape)
      #print(type(x[0].numpy()))
    i += 1
    if i==10:
      break



def load(filename):
  """テスト"""
  dataset = tf.data.TFRecordDataset(filename)
  dataset = dataset.map(parse_features)
  print(dataset)
  i = 0
  for img, mask in dataset:
    #print(type(img))
    img = byte2np(img)
    mask = byte2np(mask)
    print(img.shape, mask.shape, img.dtype, mask.dtype)
    #cv2.imwrite(f'x_{str(i)}.png', img)
    #cv2.imwrite(f'y_{str(i)}.png', mask)
    i += 1


def tf_load(filename):
  """テスト"""
  dataset = tf.data.TFRecordDataset(filename)
  dataset = dataset.map(tf_parse_features)
  print(dataset)
  print(list(dataset))
  i = 0
  for img, mask in dataset:
    img = byte2np(img)
    mask = byte2np(mask)
    if i==0:
      pass
      #print(img)
    #cv2.imwrite(f'x_{str(i)}.png', img)
    #cv2.imwrite(f'y_{str(i)}.png', mask)
    i += 1


#def _test(tra)
if __name__ == '__main__':
  load_gen()
  #main()
  #load('_record_0.tfrecord')
