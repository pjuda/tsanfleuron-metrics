import pickle

import numpy as np

import geone
import properscoring

# Scores

def crps_volume(true_field, simulations):
    true_volume = np.sum(true_field)
    volumes = np.sum(np.sum(simulations, axis=0), axis=0)
    return properscoring.crps_ensemble(true_volume, volumes)/400/400

def crps_pointwise(true_field, simulations):
    return properscoring.crps_ensemble(true_field, simulations)

def abs_mean_volume(true_field, simulations):
    true_volume = np.sum(true_field)
    volumes = np.sum(np.sum(simulations, axis=0), axis=0)
    return np.abs(true_volume - np.mean(volumes))/400/400

def abs_mean_pointwise(true_field, simulations):
    return np.abs(true_field - np.mean(simulations, axis=-1))

# Histogram measures

def mean_JS_flow(true_flow, simulations):
    p, bins = get_hist(true_flow)
    return np.mean([JS(bins, p, get_hist(simulations[...,i])[0]) for i in range(simulations.shape[-1])])

def get_hist(flow):
    bins = [0.5] + [2**x + 0.5  for x in range(0, 14)]
    hist = np.histogram((flow.reshape(-1)), bins, density=True)
    return hist

def integral(bins, values):
    return np.sum((bins[1:]-bins[:-1]) * values)

def xlogx(x):
    return np.array([0 if xi==0 else xi*np.log(xi) for xi in x])

def KL(bins, p, q):
    return integral(bins, xlogx(p)) - integral(bins, xlogx(q))

def JS(bins, p, q):
    b = np.array(bins)
    P = np.array(p)
    Q = np.array(q)
    return 0.5 * (KL(b, P, (P+Q)/2) + KL(b, Q, (P+Q)/2)) 

class Loader:
    def __init__(self, filename):
        with open(filename, 'rb') as file_handle:
            self.output = pickle.load(file_handle)
        for method in ['mps', 'krig', 'sgs']:
            if method in filename:
                self.method = method
    def get_true_dem(self):
        if self.method in ['sgs', 'krig']:
            return self.output[0][0]
        else:
            return self.output[0]
    
    def get_true_flow(self):
        if self.method in ['sgs', 'krig']:
            return self.output[0][1]
        else:
            return self.output[1]
        
    def get_simulated_dems(self):
        if self.method == 'krig':
            return np.stack([self.output[1][0]], axis=-1)
        if self.method == 'sgs':
            return np.stack(self.output[1][0], axis=-1)
        else:
            return np.stack(self.output[2][0], axis=-1)
        
    def get_simulated_flows(self):
        if self.method == 'krig':
            return np.stack([self.output[1][1]], axis=-1)
        elif self.method == 'sgs':
            return np.stack(self.output[1][1], axis=-1) 
        else:
            return np.stack(self.output[2][1], axis=-1)
    
    def get_conditioning_points(self):
        if self.method in ['sgs', 'krig']:
            return self.output[2]
        else:
            return self.output[3]
    
    def get_area(self):
        if self.method in ['sgs', 'krig']:
            return self.output[3]
        else:
            return self.output[4]
    
    def get_Img_true_dem(self):
        dem = self.get_true_dem()
        return self.get_Img(dem)
    
    def get_Img_sim_dem(self, i):
        dem = self.get_simulated_dems()[:,:,i]
        return self.get_Img(dem)

    def get_Img(self, dem):
        area = self.get_area()
        return geone.img.Img(nx=area[1]-area[0], ny=area[3]-area[2], nz=1, nv=1,
                            sx=1, sy=1, sz=1,
                            ox=area[0], oy=area[2],
                            val=np.expand_dims(dem, (0,1)),
                            varname='dem')
        