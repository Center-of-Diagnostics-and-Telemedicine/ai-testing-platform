import os
import base64
import tempfile
import pandas as pd
from metrics_lib import *

def apply_metric(metric, img1, img2):
  return metric(img1, img2)

def calc_simple_metrics(img1, img2):
  metrics_index = ['jac', 'dice', 'tpfv', 'prec', 'vs', 'jac_mod_1_8', 'jac_mod_1_4', 'jac_mod_3_8', 'jac_mod_5_8', 'jac_mod_3_4', 'jac_mod_7_8', 'num_dif', 'adj_rand',
          'nm1', 'nm2', 'nm3', 'hd', 'avvs', 'bld', 'jac_bnd', 'dice_bnd', 'tpfv_bnd', 'prec_bnd', 'vs_bnd', 'avg_dist_cnts', 'l1_edt', 'l2_edt']
  metrics = [jac, dice, tpfv, prec, vs, jac_mod_1_8, jac_mod_1_4, jac_mod_3_8, jac_mod_5_8, jac_mod_3_4, jac_mod_7_8, num_dif, adj_rand, new_metric_1, new_metric_2, new_metric_3, sym_hausdorf_cnts, avvs, bld_cnts,
            jac_bnd, dice_bnd, tpfv_bnd, prec_bnd, vs_bnd, avg_dist_cnts, l1_edt, l2_edt]
  metric_results = []

  for metric in metrics:
    metric_results.append(apply_metric(metric, img1, img2))
  return pd.Series(metric_results, index = [metrics_index])

def averaged_metrics_1(img1, img2):
  expert_img = img1
  sample_img = img2
  metrics = [jac_bnd, dice_bnd, tpfv_bnd, prec_bnd, vs_bnd, jac, dice, tpfv, prec, vs, jac_mod_1_8, jac_mod_1_4, jac_mod_3_8, jac_mod_5_8, jac_mod_3_4, jac_mod_7_8, adj_rand]
  pairs_exp_to_sample = closest_areas(expert_img, sample_img)
  pairs_sample_to_exp = closest_areas(sample_img, expert_img)
  metric_results_e_to_s = []
  metric_results_s_to_e = []
  metrics_index = ['jac_bnd_areas', 'dice_bnd_areas', 'tpfv_bnd_areas', 'prec_bnd_areas', 'vs_bnd_areas', 'jac_areas', 'dice_areas', 'tpfv_areas', 'prec_areas', 'vs_areas', 
                 'jac_mod_1_8_areas', 'jac_mod_1_4_areas', 'jac_mod_3_8_areas', 'jac_mod_5_8_areas', 'jac_mod_3_4_areas', 'jac_mod_7_8_areas', 'adj_rand_areas']
  if pairs_exp_to_sample is None:
    for metric in metrics:
      res = 0.5
      metric_results_e_to_s.append(res)
      metric_results_s_to_e.append(res)
  else:
    l_e_s = len(pairs_exp_to_sample)
    l_s_e = len(pairs_sample_to_exp)

    for metric in metrics:
      res = 0

      if l_e_s == 0:
        #if np.sum(expert_img) != 0:
        #  res = apply_metric(metric, expert_img, sample_img) * np.sum(expert_img) / (np.sum(expert_img) + np.sum(sample_img) + eps)
        #else:
        res = apply_metric(metric, expert_img, sample_img) * expert_img.sum(dtype=np.int64) / (expert_img.sum(dtype=np.int64) + sample_img.sum(dtype=np.int64) + eps)
        metric_results_e_to_s.append(res)
      else:
        for pair in pairs_exp_to_sample:
          res += apply_metric(metric, pair[0], pair[1]) * pair[0].sum(dtype=np.int64) / (expert_img.sum(dtype=np.int64) + sample_img.sum(dtype=np.int64) + eps)
        metric_results_e_to_s.append(res)

    for metric in metrics:
      res = 0
      if l_s_e == 0:
        #if np.sum(expert_img) != 0:
        #  res = apply_metric(metric, expert_img, sample_img) * np.sum(sample_img) / (np.sum(expert_img) + np.sum(sample_img) + eps)
        #else:
        res = apply_metric(metric, expert_img, sample_img) * sample_img.sum(dtype=np.int64) / (expert_img.sum(dtype=np.int64) + sample_img.sum(dtype=np.int64) + eps)
        metric_results_s_to_e.append(res)
      else:
        for pair in pairs_sample_to_exp:
          res += apply_metric(metric, pair[0], pair[1])*pair[0].sum(dtype=np.int64) / (expert_img.sum(dtype=np.int64) + sample_img.sum(dtype=np.int64) + eps)
        metric_results_s_to_e.append(res)

  return pd.Series(np.add(metric_results_e_to_s, metric_results_s_to_e), index = [metrics_index])

def averaged_metrics_0(img1, img2):
  expert_img = img1
  sample_img = img2
  metrics_index = ['hd_areas', 'avvs_areas', 'bld_areas', 'nm1_areas', 'nm2_areas', 'nm3_areas', 'avg_dist_cnts_areas', 'l1_edt_areas', 'l2_edt_areas']
  metrics = [sym_hausdorf_cnts, avvs, bld_cnts, new_metric_1, new_metric_2, new_metric_3, avg_dist_cnts, l1_edt, l2_edt]
  pairs_exp_to_sample = closest_areas(expert_img, sample_img)
  pairs_sample_to_exp = closest_areas(sample_img, expert_img)
  metric_results_e_to_s = []
  metric_results_s_to_e = []
 
  if pairs_exp_to_sample is None:
    for metric in metrics:
      res = 0
      metric_results_e_to_s.append(res)
      metric_results_s_to_e.append(res)
  else:
    l_e_s = len(pairs_exp_to_sample)
    l_s_e = len(pairs_sample_to_exp)
 
    for metric in metrics:
      res = 0
 
      if l_e_s == 0:
        res = apply_metric(metric, expert_img, sample_img) * expert_img.sum(dtype=np.int64) / (expert_img.sum(dtype=np.int64) + sample_img.sum(dtype=np.int64) + eps)
        metric_results_e_to_s.append(res)
      else:
        for pair in pairs_exp_to_sample:
          res += apply_metric(metric, pair[0], pair[1]) * pair[0].sum(dtype=np.int64) / (expert_img.sum(dtype=np.int64) + sample_img.sum(dtype=np.int64) + eps)
        metric_results_e_to_s.append(res)
 
    for metric in metrics:
      res = 0
      if l_s_e == 0:
        res = apply_metric(metric, expert_img, sample_img) * sample_img.sum(dtype=np.int64) / (expert_img.sum(dtype=np.int64) + sample_img.sum(dtype=np.int64) + eps)
        metric_results_s_to_e.append(res)
      else:
        for pair in pairs_sample_to_exp:
          res += apply_metric(metric, pair[0], pair[1])*pair[0].sum(dtype=np.int64) / (expert_img.sum(dtype=np.int64) + sample_img.sum(dtype=np.int64) + eps)
        metric_results_s_to_e.append(res)
 
  return pd.Series(np.add(metric_results_e_to_s, metric_results_s_to_e), index = [metrics_index])

def simple_averaged_metrics_0(img1, img2):
  expert_img = img1
  sample_img = img2
  metrics_index = ['hd_av_areas', 'avvs_av_areas', 'bld_av_areas', 'nm1_av_areas', 'nm2_av_areas', 'nm3_av_areas', 'avg_dist_cnts_av_areas', 'l1_edt_av_areas', 'l2_edt_av_areas']
  pairs_exp_to_sample = closest_areas(expert_img, sample_img)
  pairs_sample_to_exp = closest_areas(sample_img, expert_img)
  metric_results_e_to_s = []
  metric_results_s_to_e = []
  metrics = [sym_hausdorf_cnts, avvs, bld_cnts, new_metric_1, new_metric_2, new_metric_3, avg_dist_cnts, l1_edt, l2_edt]
  
  if pairs_exp_to_sample is None:
    for metric in metrics:
      res = 0
      metric_results_e_to_s.append(res)
      metric_results_s_to_e.append(res)
  else:
    l_e_s = len(pairs_exp_to_sample)
    l_s_e = len(pairs_sample_to_exp)

    for metric in metrics:
      res = 0
      if l_e_s == 0:
        res = apply_metric(metric, expert_img, sample_img)
        metric_results_e_to_s.append(res)
      else:
        for pair in pairs_exp_to_sample:
          res += apply_metric(metric, pair[0], pair[1])
        metric_results_e_to_s.append(res / l_e_s)

    for metric in metrics:
      res = 0
      if l_s_e == 0:
        res = apply_metric(metric, expert_img, sample_img)
        metric_results_s_to_e.append(res)
      else:
        for pair in pairs_sample_to_exp:
          res += apply_metric(metric, pair[0], pair[1])
        metric_results_s_to_e.append(res / l_s_e)
  res_metric = [max(m1, m2) for m1, m2 in zip(metric_results_s_to_e, metric_results_e_to_s)]
  return pd.Series(res_metric, index = [metrics_index])

def simple_averaged_metrics_1(img1, img2):
  expert_img = img1
  sample_img = img2
  metrics_index = ['jac_bnd_av_areas', 'dice_bnd_av_areas', 'tpfv_bnd_av_areas', 'prec_bnd_av_areas', 'vs_bnd_av_areas', 'jac_av_areas', 'dice_av_areas', 'tpfv_av_areas', 'prec_av_areas', 'vs_av_areas', 
                 'jac_mod_1_8_av_areas', 'jac_mod_1_4_av_areas', 'jac_mod_3_8_av_areas', 'jac_mod_5_8_av_areas', 'jac_mod_3_4_av_areas', 'jac_mod_7_8_av_areas', 'adj_rand_av_areas']
  pairs_exp_to_sample = closest_areas(expert_img, sample_img)
  pairs_sample_to_exp = closest_areas(sample_img, expert_img)
  metric_results_e_to_s = []
  metric_results_s_to_e = []
  metrics = [jac_bnd, dice_bnd, tpfv_bnd, prec_bnd, vs_bnd, jac, dice, tpfv, prec, vs, jac_mod_1_8, jac_mod_1_4, jac_mod_3_8, jac_mod_5_8, jac_mod_3_4, jac_mod_7_8, adj_rand]
  
  if pairs_exp_to_sample is None:
    for metric in metrics:
      res = 1
      metric_results_e_to_s.append(res)
      metric_results_s_to_e.append(res)
  else:
    l_e_s = len(pairs_exp_to_sample)
    l_s_e = len(pairs_sample_to_exp)

    for metric in metrics:
      res = 0
      if l_e_s == 0:
        res = apply_metric(metric, expert_img, sample_img)
        metric_results_e_to_s.append(res)
      else:
        for pair in pairs_exp_to_sample:
          res += apply_metric(metric, pair[0], pair[1])
        metric_results_e_to_s.append(res / l_e_s)

    for metric in metrics:
      res = 0
      if l_s_e == 0:
        res = apply_metric(metric, expert_img, sample_img)
        metric_results_s_to_e.append(res)
      else:
        for pair in pairs_sample_to_exp:
          res += apply_metric(metric, pair[0], pair[1])
        metric_results_s_to_e.append(res / l_s_e)
  res_metric = [max(m1, m2) for m1, m2 in zip(metric_results_s_to_e, metric_results_e_to_s)]
  return pd.Series(res_metric, index = [metrics_index])


def apply_all_metrics(mask_1, mask_2, return_type='dict'):
    """
    Parameters:
        mask_1, mask_2 - both are numpy.arrays with shape (n, n)
        return_type: dict, series

    Return:
        dict, pandas.Series
    """
    ser1 = calc_simple_metrics(mask_1, mask_2)
    ser2 = averaged_metrics_0(mask_1, mask_2)
    ser3 = averaged_metrics_1(mask_1, mask_2)
    ser4 = simple_averaged_metrics_0(mask_1, mask_2)
    ser5 = simple_averaged_metrics_1(mask_1, mask_2)    
    ser = ser1.append([ser2, ser3, ser4, ser5])

    result = None
    if return_type == 'dict':
        result = {k[0]: v for (k, v) in zip(ser.index, ser.values)}

    elif return_type == 'series':
        result = ser

    else:
        raise ValueError(f'return_type {return_type} is not supported yet.')

    return result


def make_csv_file(mask_1, mask_2):
    tmp = tempfile.NamedTemporaryFile(suffix='.csv').name
    res = apply_all_metrics(mask_1, mask_2, return_type='series')
    res.to_csv(tmp, index_label = 'metric', header = ['value'])
    return tmp

def write_tmp_bytes_return_tmp_path(bytes_):
    path = tempfile.NamedTemporaryFile(suffix='.png').name
    with open(file=path, mode='wb') as f:
        f.write(bytes_)
    return path

def read_bytes(path):
    data = None
    with open(file=path, mode='rb') as f:
        data = f.read()
    return data

def write_img_return_base64(arr2d):
    path = tempfile.NamedTemporaryFile(suffix='.png').name
    cv2.imwrite(path, arr2d)
    return base64.b64encode(read_bytes(path)).decode('ascii')

def get_metrics(fname1, fname2):
    mask_1 = cv2.imread(fname1,0)
    mask_1 = np.where(mask_1 > 1, 1, mask_1)
    mask_2 = cv2.imread(fname2,0)
    mask_2 = np.where(mask_2 > 1, 1, mask_2)
        
    ser1 = calc_simple_metrics(mask_1, mask_2)
    ser2 = averaged_metrics_0(mask_1, mask_2)
    ser3 = averaged_metrics_1(mask_1, mask_2)
    ser4 = simple_averaged_metrics_0(mask_1, mask_2)
    ser5 = simple_averaged_metrics_1(mask_1, mask_2)
    
    res = ser1.append([ser2, ser3, ser4, ser5])
    
    res.to_csv('metrics_res.csv', index_label = 'metric', header = ['value'])


if __name__ == "__main__":
    fname1 = '00000344_003_s1.png'
    fname2 = '00000344_003_s3.png'
    
    mask_1 = cv2.imread(fname1,0)
    mask_1 = np.where(mask_1 > 1, 1, mask_1)
    mask_2 = cv2.imread(fname2,0)
    mask_2 = np.where(mask_2 > 1, 1, mask_2)

    print(apply_all_metrics(mask_1, mask_2, return_type='dict'))
    res = apply_all_metrics(mask_1, mask_2, return_type='series')
    res.to_csv('metrics_res.csv', index_label = 'metric', header = ['value'])
