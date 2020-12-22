from sklearn.metrics.cluster import adjusted_rand_score
from scipy.spatial.distance import directed_hausdorff, cdist
from scipy import ndimage
import cv2 #version 4+
import numpy as np

eps = 1e-6
def jac(expert, sample):
  intersection = cv2.bitwise_and(expert,sample)
  union = cv2.bitwise_or(expert,sample)
  if union.sum(dtype=np.int64) == 0:
    return 1
  return intersection.sum(dtype=np.int64) / (union.sum(dtype=np.int64) + eps)

def dice(expert, sample):
  intersection = cv2.bitwise_and(expert,sample)
  if (expert.sum(dtype=np.int64) + sample.sum(dtype=np.int64)) == 0:
    return 1
  return (2*intersection.sum(dtype=np.int64)
          ) / (expert.sum(dtype=np.int64) + sample.sum(dtype=np.int64) + eps)

def tpfv(expert, sample):
  intersection = cv2.bitwise_and(expert,sample)
  if expert.sum(dtype=np.int64) == 0 and sample.sum(dtype=np.int64) == 0:
    return 1
  return (intersection.sum(dtype=np.int64)) / (expert.sum(dtype=np.int64) + eps)

def prec(expert, sample):
  intersection = cv2.bitwise_and(expert,sample)
  if expert.sum(dtype=np.int64) == 0 and sample.sum(dtype=np.int64) == 0:
    return 1
  return intersection.sum(dtype=np.int64) / (sample.sum(dtype=np.int64) + eps)

def vs(expert, sample):
  return abs(1 - (abs(sample.sum(dtype=np.int64) - expert.sum(dtype=np.int64))) / (sample.sum(dtype=np.int64) + expert.sum(dtype=np.int64) + eps))

def rvd(expert, sample):
  return abs((sample.sum(dtype=np.int64) - expert.sum(dtype=np.int64)) / (expert.sum(dtype=np.int64) + eps))

###

def get_cnt_img(bin_mask, r = 30):
  if r != 1:
    r = int(len(bin_mask) / 34)
  img_8bit = np.uint8(bin_mask * 255)
  contours, hier = cv2.findContours(img_8bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  dim = len(bin_mask)
  cnt_img = np.zeros((dim, dim))
  for i, cnt in enumerate(contours):
    cv2.drawContours(cnt_img, [cnt], -1, 1, r)
  return cnt_img

#similiraty boundary-to-area based metrics
r = 30 #width of boundary
# 5 как-то очень мало было, прям очень, кажется надо сильно больше

def jac_bnd(expert, sample):
  sample_cnt = get_cnt_img(sample, r)
  expert_cnt = get_cnt_img(expert, r)
  intersection = cv2.bitwise_and(expert_cnt,sample_cnt)
  union = cv2.bitwise_or(expert_cnt,sample_cnt)
  if union.sum(dtype=np.int64) == 0:
    return 1
  return (intersection.sum(dtype=np.int64)) / (union.sum(dtype=np.int64) + eps)

def dice_bnd(expert, sample):
  sample_cnt = get_cnt_img(sample, r)
  expert_cnt = get_cnt_img(expert, r)
  intersection = cv2.bitwise_and(expert_cnt,sample_cnt)
  if (expert_cnt.sum(dtype=np.int64) + sample_cnt.sum(dtype=np.int64)) == 0:
    return 1
  return (2*intersection.sum(dtype=np.int64)
          ) / (expert_cnt.sum(dtype=np.int64) + sample_cnt.sum(dtype=np.int64) + eps)

def tpfv_bnd(expert, sample):
  sample_cnt = get_cnt_img(sample, r)
  expert_cnt = get_cnt_img(expert, r)
  intersection = cv2.bitwise_and(expert_cnt,sample_cnt)
  if expert_cnt.sum(dtype=np.int64) == 0 and sample_cnt.sum(dtype=np.int64) == 0:
    return 1
  return (intersection.sum(dtype=np.int64)) / (expert_cnt.sum(dtype=np.int64) + eps)

def prec_bnd(expert, sample):
  sample_cnt = get_cnt_img(sample, r)
  expert_cnt = get_cnt_img(expert, r)
  intersection = cv2.bitwise_and(expert_cnt,sample_cnt)
  if expert_cnt.sum(dtype=np.int64) == 0 and sample_cnt.sum(dtype=np.int64) == 0:
    return 1
  return (intersection.sum(dtype=np.int64)) / (sample_cnt.sum(dtype=np.int64) + eps)

def vs_bnd(expert, sample):
  sample_cnt = get_cnt_img(sample, r)
  expert_cnt = get_cnt_img(expert, r)
  return abs(1 - (abs(sample_cnt.sum(dtype=np.int64) - expert_cnt.sum(dtype=np.int64))) / (sample_cnt.sum(dtype=np.int64) + expert_cnt.sum(dtype=np.int64) + eps))
#difference area based mertrics

def rvd_bnd(expert, sample):
  sample_cnt = get_cnt_img(sample, r)
  expert_cnt = get_cnt_img(expert, r)
  return abs((sample_cnt.sum(dtype=np.int64) - expert_cnt.sum(dtype=np.int64)) / (expert_cnt.sum(dtype=np.int64) + eps))

#similiraty boundary based metrics
def get_nonzero_coords(img):
  coords = np.nonzero(img)
  return [[coords[0][i], coords[1][i]] for i in range(len(coords[0]))]

def sym_hausdorf_areas(sample, expert):
  s_e = directed_hausdorff(get_nonzero_coords(sample), get_nonzero_coords(expert))[0]
  e_s = directed_hausdorff(get_nonzero_coords(expert), get_nonzero_coords(sample))[0]
  return max(abs(s_e), abs(e_s))

def sym_hausdorf_cnts(sample, expert):
  sample_cnt = get_cnt_img(sample, 1)
  expert_cnt = get_cnt_img(expert, 1)
  sample_cnt_nonzero = get_nonzero_coords(sample_cnt)
  expert_cnt_nonzero = get_nonzero_coords(expert_cnt)
  if len(expert_cnt_nonzero) == 0 and len(sample_cnt_nonzero) == 0:
    return 0
  if len(expert_cnt_nonzero) == 0 or len(sample_cnt_nonzero) == 0:
    return len(sample)
  s_e = directed_hausdorff(sample_cnt_nonzero, expert_cnt_nonzero)[0]
  e_s = directed_hausdorff(expert_cnt_nonzero, sample_cnt_nonzero)[0]
  return max(abs(s_e), abs(e_s))

def bld_cnts(sample, expert):
  sample_cnt = get_cnt_img(sample, 1)
  expert_cnt = get_cnt_img(expert, 1)
  coords_s = get_nonzero_coords(sample_cnt)
  coords_e = get_nonzero_coords(expert_cnt)
  if len(coords_e) == 0 and len(coords_s) == 0:
    return 0
  if len(coords_e) == 0 or len(coords_s) == 0:
    return len(sample)
  
  fmind = len(sample)
  fmaxd = 0
  for ref_point in coords_s:
    ref_point_to_expert = cdist([ref_point], coords_e, 'euclidean')
    fmind = min(min(ref_point_to_expert[0]), fmind)

  for test_point in coords_e:
    test_point_to_sample = cdist([ref_point], coords_s, 'euclidean')
    fmaxd = max(min(test_point_to_sample[0]), fmaxd)

  #fmind = min(ref_point_to_expert[0])
  #fmaxd = max(test_point_to_sample[0])
  return max(fmind, fmaxd)

def avvs(sample, expert):
  sample_cnt = get_cnt_img(sample, 1)
  expert_cnt = get_cnt_img(expert, 1)
  coords_s = get_nonzero_coords(sample_cnt)
  coords_e = get_nonzero_coords(expert_cnt)
  if len(coords_e) == 0 and len(coords_s) == 0:
    return 0
  if len(coords_e) == 0 or len(coords_s) == 0:
    return len(sample)
  mm = cdist(coords_s, coords_e, 'euclidean')
  totallen = len(np.concatenate(mm))
  totalsum = np.concatenate(mm).sum(dtype=np.int64)
  return totalsum / totallen
  
def jac_mod_1_8(expert, sample):
  a = 1/8
  intersection = cv2.bitwise_and(expert,sample)
  union = cv2.bitwise_or(expert,sample)
  fp = abs(sample.sum(dtype=np.int64) - union.sum(dtype=np.int64))
  fn = abs(expert.sum(dtype=np.int64) - union.sum(dtype=np.int64))
  if union.sum(dtype=np.int64) == 0:
    return 1
  return (intersection.sum(dtype=np.int64)) / (intersection.sum(dtype=np.int64) + eps + 2*(1-a)*fp + 2*a*fn)

def jac_mod_3_8(expert, sample):
  a = 3/8
  intersection = cv2.bitwise_and(expert,sample)
  union = cv2.bitwise_or(expert,sample)
  fp = abs(sample.sum(dtype=np.int64) - union.sum(dtype=np.int64))
  fn = abs(expert.sum(dtype=np.int64) - union.sum(dtype=np.int64))
  if union.sum(dtype=np.int64) == 0:
    return 1
  return (intersection.sum(dtype=np.int64)) / (intersection.sum(dtype=np.int64) + eps + 2*(1-a)*fp + 2*a*fn)

def jac_mod_5_8(expert, sample):
  a = 5/8
  intersection = cv2.bitwise_and(expert,sample)
  union = cv2.bitwise_or(expert,sample)
  fp = abs(sample.sum(dtype=np.int64) - union.sum(dtype=np.int64))
  fn = abs(expert.sum(dtype=np.int64) - union.sum(dtype=np.int64))
  if union.sum(dtype=np.int64) == 0:
    return 1
  return (intersection.sum(dtype=np.int64)) / (intersection.sum(dtype=np.int64) + eps + 2*(1-a)*fp + 2*a*fn)

def jac_mod_7_8(expert, sample):
  a = 7/8
  intersection = cv2.bitwise_and(expert,sample)
  union = cv2.bitwise_or(expert,sample)
  fp = abs(sample.sum(dtype=np.int64) - union.sum(dtype=np.int64))
  fn = abs(expert.sum(dtype=np.int64) - union.sum(dtype=np.int64))
  if union.sum(dtype=np.int64) == 0:
    return 1
  return (intersection.sum(dtype=np.int64)) / (intersection.sum(dtype=np.int64) + eps + 2*(1-a)*fp + 2*a*fn)

def jac_mod_1_4(expert, sample):
  a = 1/4
  intersection = cv2.bitwise_and(expert,sample)
  union = cv2.bitwise_or(expert,sample)
  fp = abs(sample.sum(dtype=np.int64) - union.sum(dtype=np.int64))
  fn = abs(expert.sum(dtype=np.int64) - union.sum(dtype=np.int64))
  if union.sum(dtype=np.int64) == 0:
    return 1
  return (intersection.sum(dtype=np.int64)) / (intersection.sum(dtype=np.int64) + eps + 2*(1-a)*fp + 2*a*fn)

def jac_mod_3_4(expert, sample):
  a = 3/4
  intersection = cv2.bitwise_and(expert,sample)
  union = cv2.bitwise_or(expert,sample)
  fp = abs(sample.sum(dtype=np.int64) - union.sum(dtype=np.int64))
  fn = abs(expert.sum(dtype=np.int64) - union.sum(dtype=np.int64))
  if union.sum(dtype=np.int64) == 0:
    return 1
  return (intersection.sum(dtype=np.int64)) / (intersection.sum(dtype=np.int64) + eps + 2*(1-a)*fp + 2*a*fn)

#новые метрики от Дениса

a = 1/7
def jac_mod(expert, sample, a = 1/7):
  intersection = cv2.bitwise_and(expert,sample)
  union = cv2.bitwise_or(expert,sample)
  fp = abs(sample.sum(dtype=np.int64) - union.sum(dtype=np.int64))
  fn = abs(expert.sum(dtype=np.int64) - union.sum(dtype=np.int64))
  if union.sum(dtype=np.int64) == 0:
    return 1
  return (intersection.sum(dtype=np.int64)) / (intersection.sum(dtype=np.int64) + eps + 2*(1-a)*fp + 2*a*fn)

import numpy.ma as ma 
import math 
#my_nparray = np.zeros((1024,1024))

def square_distance(centroid_coords, point):
    return ((point[0] - centroid_coords[0])**2 + (point[1] - centroid_coords[1])**2)
    
def average_distance_count(image, expert):
    image_coords = (np.argwhere( image) ) #переводим бинарную маску в массив из координат
    if (image_coords.size <= 1): # если одна точка, то среднее расстояние есть ноль
        return 0
    else:
        average_dist_from_center_image = np.var(image_coords, axis=0) #средне-квадратичные расстояния до центра по координатам x и y
        average_distance_image =  2*(image_coords.size/(image_coords.size - 1)) * (average_dist_from_center_image[0] + average_dist_from_center_image[1] )
        return average_distance_image

        
def count_distance_between_sets(expert, sample):
    expert_coords = (np.argwhere( expert) ) #переводим бинарную маску в массив из координат
    sample_coords = (np.argwhere( sample) ) #переводим бинарную маску в массив из координат
    if ( expert_coords.size >= sample_coords.size ):
        big_array_coords = expert_coords
        small_array_coords = sample_coords
    else: 
        big_array_coords = sample_coords
        small_array_coords = expert_coords
    #big_array_coords = (np.argwhere( big_array) ) #переводим бинарную маску в массив из координат 
    if (big_array_coords.size == 0):
        return 0
    elif (big_array_coords.size == 1):
        if (np.array_equal(big_array_coords, small_array_coords) == True):
            return 0
        else:
            return 1
    else:
        centroid_big_array = np.mean(big_array_coords, axis=0) # вычислим центроид фигуры из big_array
        average_dist_from_center_big_array = np.var(big_array_coords, axis=0) #средне-квадратичные расстояния до центра по координатам x и y
        average_distance_big_array =  2*(big_array_coords.size/(big_array_coords.size - 1)) * (average_dist_from_center_big_array[0] + average_dist_from_center_big_array[1] )

        #small_array_coords = (np.argwhere( small_array) ) #переводим бинарную маску в массив из координат 
        if (small_array_coords.size == 0):
            return average_distance_big_array
        else:
            ratio_big_to_small = round(big_array_coords.size / small_array_coords.size, 0) #отношение размера большого массива к маленькому
            #print("ratio_big_to_small = ", ratio_big_to_small)

            
            centroid_small_array = np.mean(small_array_coords, axis=0) # вычислим центроид фигуры из small_array
            average_dist_from_center_small_array = np.var(small_array_coords, axis=0) #средне-квадратичные расстояния до центра по координатам x и y
            if (small_array_coords.size == 1):
                average_distance_small_array = 0
            else:
                average_distance_small_array =  2*(small_array_coords.size/(small_array_coords.size - 1)) * (average_dist_from_center_small_array[0] + average_dist_from_center_small_array[1] )
        
            #вычислили все параметры для исходных массивов. Теперь посмотрим на Z = big_array + ratio_big_to_small*small_array
            centroid_Z = (centroid_big_array + ratio_big_to_small*centroid_small_array)/(ratio_big_to_small + 1)
            #print("centroid_Z = ", centroid_Z)

            #print("square_distance( centroid_Z, small_array_coords ) = ", square_distance( centroid_Z, small_array_coords ))
            #print("small_array_coords = ", small_array_coords)
            #squares = np.array([squarer(xi) for xi in x])

            average_distance_small_array_from_center_of_Z = np.average(np.array([square_distance(centroid_Z, point) for point in small_array_coords ])) # square_distance( centroid_Z, small_array_coords )/( small_array_coords.size )
            #print("average_distance_small_array_from_center_of_Z = ", average_distance_small_array_from_center_of_Z)
            average_distance_big_array_from_center_of_Z = np.average(np.array([square_distance(centroid_Z, point) for point in big_array_coords ]))
            #print("average_distance_big_array_from_center_of_Z = ", average_distance_big_array_from_center_of_Z)
            
            average_distance_Z = 2*(( big_array_coords.size + ratio_big_to_small * small_array_coords.size )/( big_array_coords.size + ratio_big_to_small * small_array_coords.size - 1)) * ( big_array_coords.size * average_distance_big_array_from_center_of_Z + ratio_big_to_small * small_array_coords.size * average_distance_small_array_from_center_of_Z)/( big_array_coords.size + ratio_big_to_small * small_array_coords.size ) #среднее квадратичное расстояние между вершинами Z
            #print("average_distance_Z = ", average_distance_Z)
            average_distance_between_sets_square = 0.5 * ( ( big_array_coords.size + ratio_big_to_small * small_array_coords.size )* (big_array_coords.size + ratio_big_to_small * small_array_coords.size - 1) * average_distance_Z - big_array_coords.size * (big_array_coords.size - 1) * average_distance_big_array - ratio_big_to_small * small_array_coords.size * (ratio_big_to_small * small_array_coords.size - 1) * average_distance_small_array )/(big_array_coords.size * ratio_big_to_small * small_array_coords.size )

            return average_distance_between_sets_square


epsilon = 10**(-9)

def new_metric_1(expert, sample):
    expert_coords = (np.argwhere( expert) ) #переводим бинарную маску в массив из координат
    sample_coords = (np.argwhere( sample) ) #переводим бинарную маску в массив из координат
    if (expert_coords.size == 0) and (sample_coords.size == 0):
        return 0 #всё хорошо, множества сильно близки
    elif (expert_coords.size == 0) or (sample_coords.size == 0):
        return 1 #одна бинарная маска -- пустая, другая -- нет
    else:
        if ( expert_coords.size >= sample_coords.size ):
            big_array = expert
            small_array = sample
        else: 
            big_array = sample
            small_array = expert
        result = 1 -  (math.sqrt(average_distance_count(big_array, expert)) + math.sqrt( average_distance_count(small_array, expert)))/(2*math.sqrt(count_distance_between_sets(big_array, small_array)))
        return abs(result)


def new_metric_2(expert, sample):
    expert_coords = (np.argwhere( expert) ) #переводим бинарную маску в массив из координат
    sample_coords = (np.argwhere( sample) ) #переводим бинарную маску в массив из координат
    if (expert_coords.size == 0) and (sample_coords.size == 0):
        return 0 #всё хорошо, множества сильно близки
    elif (expert_coords.size == 0) or (sample_coords.size == 0):
        return 1 #одна бинарная маска -- пустая, другая -- нет
    else:
        if ( expert_coords.size >= sample_coords.size ):
            big_array = expert
            small_array = sample 
        else: 
            big_array = sample
            small_array = expert
        result = 1 - math.sqrt( (average_distance_count(big_array, expert) * ( average_distance_count(small_array, expert) ) )/(count_distance_between_sets(big_array, small_array)**2))
        return abs(result)


def new_metric_3(expert, sample):
    expert_coords = (np.argwhere( expert) ) #переводим бинарную маску в массив из координат
    sample_coords = (np.argwhere( sample) ) #переводим бинарную маску в массив из координат
    if (expert_coords.size == 0) and (sample_coords.size == 0):
        return 0 #всё хорошо, множества сильно близки
    elif (expert_coords.size == 0) or (sample_coords.size == 0):
        return 1 #одна бинарная маска -- пустая, другая -- нет
    else:
        if ( expert_coords.size >= sample_coords.size ):
            big_array = expert
            small_array = sample
        else: 
            big_array = sample
            small_array = expert
        result = 1 - math.sqrt(math.sqrt( (average_distance_count(big_array, expert) * ( average_distance_count(small_array, expert) ) )/(count_distance_between_sets(big_array, small_array)**2)))
        return abs(result)
        
def avg_dist_cnts(sample, expert):
  sample_cnt = get_cnt_img(sample, 1)
  expert_cnt = get_cnt_img(expert, 1)
  coords_s = get_nonzero_coords(sample_cnt)
  coords_e = get_nonzero_coords(expert_cnt)
  if len(coords_e) == 0 and len(coords_s) == 0:
    return 0
  if len(coords_e) == 0 or len(coords_s) == 0:
    return len(expert)
  
  fmind = len(expert)
  fmaxd = 0
  sum_s_e = 0
  sum_e_s = 0
  for ref_point in coords_s:
    ref_point_to_expert = cdist([ref_point], coords_e, 'euclidean')
    sum_s_e += min(ref_point_to_expert[0])

  for test_point in coords_e:
    test_point_to_sample = cdist([ref_point], coords_s, 'euclidean')
    sum_e_s += min(test_point_to_sample[0])

  return (0.5 * (sum_s_e / len(coords_s) + sum_e_s / len(coords_e)))

def adj_rand(expert, sample):
  intersection = cv2.bitwise_and(expert,sample)
  union = cv2.bitwise_or(expert,sample)
  if union.sum(dtype=np.int64) == 0:
    return 1
  if intersection.sum(dtype=np.int64) == 0:
    return -1

  expert_1d = expert.flatten()
  sample_1d = sample.flatten()
  return (adjusted_rand_score(expert_1d, sample_1d) + 1) / 2

#разность кол-ва связных контуров - ток для масок в целом
def num_dif(sample, expert):
  contours_s, hier_s = cv2.findContours(sample, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours_e, hier_e = cv2.findContours(expert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  return abs(len(contours_e) - len(contours_s))

def l1_edt(sample, expert):
  union = cv2.bitwise_or(expert,sample)
  if union.sum(dtype=np.int64) == 0:
    return 0
  sample_edt = ndimage.distance_transform_edt(sample)
  expert_edt = ndimage.distance_transform_edt(expert)
  return np.absolute(sample_edt-expert_edt).sum(dtype=np.int64) / union.sum(dtype=np.int64)

def l2_edt(sample, expert):
  union = cv2.bitwise_or(expert,sample)
  if union.sum(dtype=np.int64) == 0:
    return 0
  sample_edt = ndimage.distance_transform_edt(sample)
  expert_edt = ndimage.distance_transform_edt(expert)
  return math.sqrt(((sample_edt-expert_edt)**2).sum(dtype=np.int64) / union.sum(dtype=np.int64))
  
#разбиение по ближайшим парам

def get_areas_and_coords(img):
  img_8bit = np.uint8(img * 255)
  contours, hier = cv2.findContours(img_8bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  areas = []
  c_coords = []
  for i, cnt in enumerate(contours):
      dim = len(img)
      cnt_img = np.zeros((dim, dim))
      M = cv2.moments(cnt)
      if len(cnt) == 2:
          break
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
      cv2.drawContours(cnt_img, [cnt], -1, 1, -1)
      areas.append(cnt_img)
      c_coords.append((cX, cY))
  return areas, c_coords

def closest_areas(img1, img2):
  #img1 - expert - из него берем области и ищем под них подходящие
  areas1, coords1 = get_areas_and_coords(img1)
  areas2, coords2 = get_areas_and_coords(img2)
  area_pairs = []
  if len(coords2) == 0 and len(coords1) == 0:
    return None
  elif len(coords2) == 0 or len(coords1) == 0:
    return area_pairs
  else:
    for i, center in enumerate(coords1):
      closest_index = cdist([center], coords2).argmin()
      area_pairs.append((areas1[i], areas2[closest_index]))
    return area_pairs
