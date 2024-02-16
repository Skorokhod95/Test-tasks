from PointNet import get_shape_segmentation_model as pointnet_seg
import keras
import os
import numpy as np
from tqdm import tqdm

def get_data(dir):
    clouds = []
    clouds_names = os.listdir(dir)
    for cloud in clouds_names:
        clouds.append(np.fromfile(dir+'/'+cloud, np.float32).reshape(-1,5))
    return clouds, clouds_names

class DataGenerator(keras.utils.Sequence):

    def __init__(self, data, labels, batch_size=32, shuffle=True, npoints=500, class_weight=None):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        self.npoints = npoints
        self.nclasses = 3
        self.class_weight = class_weight
        self.reminder = len(self.data) % batch_size
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        n = len(self.data) // self.batch_size
        if self.reminder != 0:
           return n+1
        return n

    def __getitem__(self, index):
        if index+1 == self.__len__() and self.reminder != 0:
          indexes = self.indexes[index * self.batch_size:]
        else:
          indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        data = list(map(lambda x: self.cut([self.data[x], self.labels[x]]), indexes))

        if self.class_weight:
          batch_data, batch_labels, batsh_sample_weights = [list(x) for x in zip(*data)]
          return np.array(batch_data), np.array(batch_labels), np.array(batsh_sample_weights)

        batch_data, batch_labels = [list(x) for x in zip(*data)]
        return np.array(batch_data), np.array(batch_labels)

    def cut(self, sample):

        points, labels = sample

        #cut road and background behind of the lidar
        x = points[:, 0].reshape(points.shape[0])
        y = points[:, 1].reshape(points.shape[0])

        condition = (y>-20)&(y<20)&(x>-5)&(x<40)

        points, labels = points[condition], labels[condition]

        #choose npoints number of points from cloud
        index = np.arange(len(points))

        return self.choose_points(points, labels)


    def choose_points(self, points, labels):
      N = len(labels)
      index = np.arange(N)
      new_index = np.array([])
      LABELS = list(set(labels))

      labels_transformed = np.zeros((N, self.nclasses))
      for i in LABELS:
        labels_transformed[labels == i, i] = 1

      if N >= self.npoints:
        for i in LABELS:
          if i != 0:
            new_index = np.append(new_index, index[labels == i])
        N0 = self.npoints - len(new_index)
        new_index = np.append(new_index, np.random.choice(index[labels == 0], N0))
      else:
        p = N / self.npoints
        for i in LABELS:
          j = index[labels == i]
          new_index = np.append(new_index, j)
          new_index = np.append(new_index, np.random.choice(j, int(p * len(j))+1), replace=True)
        new_index = np.random.choice(new_index, self.npoints)

      new_index = new_index.astype(np.int32)

      if self.class_weight:
        sw = self.get_sample_weights(labels[new_index])
        return points[new_index][:, :3], labels_transformed[new_index], sw

      return points[new_index][:, :3], labels_transformed[new_index]


    def get_sample_weights(self, labels):
      return np.array(list(map(lambda y: self.class_weight.get(int(y), 1.0), labels)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def main():
    os.chdir('C:\\Users\\denet\\OneDrive\\Документы\\Alina\\Поиск работы\\Тестовое ФГУП НАМИ\\Code')
    data, names = get_data(os.getcwd()+'/test')
    dg = DataGenerator(data, [np.array([0] * len(i)) for i in data], npoints=500)

    model = pointnet_seg(num_points=500, num_classes=3)
    model.load_weights('best.weights.h5')

    predictions = model.predict(dg)
    colors = np.zeros(predictions.shape)
    for clas in range(3):
        colors[:, :, clas] = (np.argmax(predictions, axis=-1) == clas).astype(np.int32)



    for n, i in tqdm(enumerate(names)):
        np.savetxt(os.getcwd()+'/test/'+i[:-3]+'label', colors[n])


if __name__ == "__main__":
    main()


