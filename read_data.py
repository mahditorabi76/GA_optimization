import numpy as np
import matplotlib.pyplot as plt
import random

# H = 0
# W = 0
# K = 0
#
# list_ni = []
# list_price = []
# list_radius = []

def get_par_in_file():
  '''
  :input: nothing
  :return: H, W, K, list_ni, list_price, list_radius
  '''
  f = open("input_section_1.txt", "r")
  l1 = f.readline().strip().split(" ")
  H = int(l1[0])
  W = int(l1[1])
  K = int(f.readline().strip().split(" ")[0])
  list_ni = []
  list_price = []
  list_radius = []
  for l in f:
    line = l.strip().split(" ")
    list_ni.append(int(line[0]))
    for i in range(int(line[0])):
      list_price.append(int(line[1]))
      list_radius.append(int(line[2]))
    # l.strip().split(" ")
    # l.strip().split(" ")
  f = open("input_section_2.txt", "r")
  list_position = []
  list_r = []
  for l in f:
    line = l.strip().split(" ")
    list_position.append([int(line[0]), int(line[1]), 1])
    list_r.append(int(line[2]))

  return H, W, K, list_ni, list_price, list_radius, list_position, list_r

def get_par_in_file2():
  '''
    :input: nothing
    :return: H, W, K, list_ni, list_price, list_radius
    '''
  f = open("input_section_2.txt", "r")
  list_position = []
  list_radius = []
  for l in f:
    line = l.strip().split(" ")
    list_position.append([int(line[0]), int(line[1]), 1])
    list_radius.append(int(line[2]))
    # l.strip().split(" ")
    # l.strip().split(" ")
  return list_position, list_radius

if __name__ == '__main__':

  H, W, K, list_ni, list_price, list_radius, list_p, list_r = get_par_in_file()
  print(len(list_ni))
  z = np.zeros((H, W))
  print(H)
  print(W)

  l1 , l2 = get_par_in_file2()
  pass
  # print(sum(list_ni))
  # n = np.random.randint(0, sum(list_ni))
  # # print(n)
  # for i in range(n):
  #   x = np.random.randint(0, H)
  #   y = np.random.randint(0, W)
  #   z[x][y] = np.random.randint(0, K)
  #
  # a = np.where(z != 0)
  # # print(a)
  # list_R = []
  # list_type = []
  # for i in range(len(a[0])):
  #   list_type.append(int(z[a[0][i]][a[1][i]]))
  #   list_R.append(list_radius[int(z[a[0][i]][a[1][i]])])
  # print(z[a[0][0]][a[1][0]])
  # print(list_R)
  #
  # fig, ax = plt.subplots()
  #
  # plt.rc('text', usetex=True)
  # plt.rc('font', family='serif')
  # # plt.plot(a[0], a[1], marker='.', color='k', linestyle='None')
  # plt.axis([0, H, 0, W])
  # plt.title('W1 galaxy cluster field')
  # plt.xlabel(r'$\vartheta$ (arcmins)')
  # plt.ylabel(r'$\vartheta$ (arcmins)')
  # list_color = ['r', 'b', 'y']
  # plt.axis("equal")
  # for i in range(len(list_R)):
  #   circle1 = plt.Circle((a[0][i], a[1][i]), list_R[i], color=random.choice(list_color)[0])
  #   # print(str(list_type[i]))
  #
  #   plt.gca().add_artist(circle1)
  #
  #
  #
  # plt.show()

  # unique, counts = np.unique(z, return_counts=True)
  # print(dict(zip(unique, counts)))

