use crate::radonn::nn::net::SeqNet;

const EPS: f32 = 1e-8f32;

pub fn feed_forward(input: &Vec<f32>, net: &SeqNet) -> Vec<f32> {
    let mut out = net.layers[0].feed_forward(input);
    for layer in 1..net.layers.len() {
        out = net.layers[layer].feed_forward(&out);
    }
    return out;
}

pub fn back_propagate<T: Optimizer>(input: &Vec<f32>, expected_out: &Vec<f32>, net: &mut SeqNet, optim: &mut Box<T>) {
    let mut out = net.layers[0].d_feed_forward(input);

    let mut cache = Vec::with_capacity(net.layers.len());
    let mut ins = Vec::with_capacity(net.layers.len());
    ins.push(input.clone());
    cache.push(out[1].clone());
    for layer in 1..net.layers.len() {
        ins.push(out[0].clone());
        out = net.layers[layer].d_feed_forward(&out[0]);
        cache.push(out[1].clone());
    }

    let mut out_err: Vec<f32> = Vec::with_capacity(expected_out.len());
    for i in 0..expected_out.len() {
        out_err.push(out[0][i] - expected_out[i]);
    }

    let mut errors: Vec<Vec<f32>> = Vec::with_capacity(net.layers.len());

    for layer in 0..net.layers.len() {
        let i = net.layers.len() - layer - 1;
        errors.push(net.layers[i].calc_err(&out_err, &ins[i]));
        out_err = net.layers[i].back_propagate(&out_err, &cache[i]);
    }
    optim.reversed_update(&errors);
}

pub fn loss(ins: &Vec<Vec<f32>>, expected_outs: &Vec<Vec<f32>>, net: &SeqNet) -> f32 {
    let mut error = 0f32;
    for i in 0..ins.len() {
        let output = feed_forward(&ins[i], net);
        for j in 0..output.len() {
            error += (output[j] - expected_outs[i][j]).powi(2);
        }
    }
    return error;
}

pub fn step<T: Optimizer>(net: &mut SeqNet, optim: &mut Box<T>) {
    let steps = optim.get_step();
    for layer in 0..net.layers.len() {
        net.layers[layer].step(&steps[layer]);
    }
}

pub trait Optimizer {
    fn reversed_update(&mut self, errors: &Vec<Vec<f32>>);

    fn get_step(&mut self) -> Vec<Vec<f32>>;
}

pub struct SGD {
    lr: f32,
    cache: Vec<Vec<f32>>,
}

pub struct SGDwMomentum {
    lr: f32,
    mom0: f32,
    mom1: f32,
    cache: Vec<Vec<f32>>,
    mom_cache: Vec<Vec<f32>>,
}

pub struct RMSProp {
    lr: f32,
    beta0: f32,
    beta1: f32,
    cache0: Vec<Vec<f32>>,
    cache1: Vec<Vec<f32>>,
}

impl SGD {
    pub fn new(net: &SeqNet, lr: f32) -> Self {
        let mut cache = Vec::with_capacity(net.layers.len());
        for i in 0..net.layers.len() {
            cache.push(vec![0.0f32; net.layers[i].param_num()]);
        }
        return Self {
            lr,
            cache,
        };
    }
}

impl SGDwMomentum {
    pub fn new(net: &SeqNet, lr: f32, mom: f32) -> Self {
        let mut cache = Vec::with_capacity(net.layers.len());
        for i in 0..net.layers.len() {
            cache.push(vec![0.0f32; net.layers[i].param_num()]);
        }
        return Self {
            mom0: mom,
            mom1: 1f32 - mom,
            lr,
            mom_cache: cache.clone(),
            cache,
        };
    }
}

impl RMSProp {
    pub fn new(net: &SeqNet, lr: f32, beta: f32) -> Self {
        let mut cache = Vec::with_capacity(net.layers.len());
        for i in 0..net.layers.len() {
            cache.push(vec![0.0f32; net.layers[i].param_num()]);
        }
        return Self {
            beta0: beta,
            beta1: 1f32 - beta,
            lr,
            cache0: cache.clone(),
            cache1: cache,
        };
    }
}


impl Optimizer for SGD {
    fn reversed_update(&mut self, errors: &Vec<Vec<f32>>) {
        for i in 0..errors.len() {
            let act_ind = errors.len() - i - 1;
            for (cache_val, err) in self.cache[i].iter_mut().zip(errors[act_ind].iter()) {
                *cache_val += err * self.lr;
            }
        }
    }

    fn get_step(&mut self) -> Vec<Vec<f32>> {
        let ret_cache = self.cache.clone();
        for i in 0..self.cache.len() {
            self.cache[i] = vec![0.0f32; self.cache[i].len()];
        }
        return ret_cache;
    }
}

impl Optimizer for SGDwMomentum {
    fn reversed_update(&mut self, errors: &Vec<Vec<f32>>) {
        for i in 0..errors.len() {
            let act_ind = errors.len() - i - 1;
            for (cache_val, err) in self.cache[i].iter_mut().zip(errors[act_ind].iter()) {
                *cache_val += err * self.lr;
            }
        }
    }

    fn get_step(&mut self) -> Vec<Vec<f32>> {
        for i in 0..self.mom_cache.len() {
            for (mc, m) in self.mom_cache[i].iter_mut().zip(self.cache[i].iter()) {
                *mc = *mc * self.mom0 + m * self.mom1;
            }
        }
        let ret_cache = self.mom_cache.clone();
        for i in 0..self.cache.len() {
            self.cache[i] = vec![0.0f32; self.cache[i].len()];
        }
        return ret_cache;
    }
}

impl Optimizer for RMSProp {
    fn reversed_update(&mut self, errors: &Vec<Vec<f32>>) {
        for i in 0..errors.len() {
            let act_ind = errors.len() - i - 1;
            for (cache_val, err) in self.cache0[i].iter_mut().zip(errors[act_ind].iter()) {
                *cache_val += err;
            }
        }
    }

    fn get_step(&mut self) -> Vec<Vec<f32>> {
        for i in 0..self.cache1.len() {
            for (c1, c0) in self.cache1[i].iter_mut().zip(self.cache0[i].iter()) {
                *c1 = *c1 * self.beta0 + c0 * c0 * self.beta1;
            }
        }
        let mut ret_val = Vec::with_capacity(self.cache0.len());

        for i in 0..self.cache0.len() {
            ret_val.push(Vec::with_capacity(self.cache0[i].len()));
            for (c0, c1) in self.cache0[i].iter().zip(self.cache1[i].iter()) {
                ret_val[i].push(*c0 / (c1.sqrt() + EPS) * self.lr);
            }
        }
        for i in 0..self.cache0.len() {
            self.cache0[i] = vec![0.0f32; self.cache0[i].len()];
        }
        return ret_val;
    }
}