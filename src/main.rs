use radonn::nn::net::{Dense, Layer, Activation, Bias, ReLU};
use radonn::nn::optim::{feed_forward, SGD};
use rand::{Rng, thread_rng};
use crate::radonn::nn::net::{Sigmoid, Softmax};
use crate::radonn::nn::optim::{SGDwMomentum, RMSProp};
use std::alloc::System;
use std::time::Instant;
use std::{io, mem, thread, time};
use std::fmt::Display;
use std::ops::Div;

mod radonn;

fn main() {
    use radonn::nn::net::SeqNet;
    use radonn::nn::optim::feed_forward;

    let layer0 = Dense {
        connections: radonn::nn::net::rand_range_xavier(2, 5),
        in_num: 2,
        out_num: 5,
    };
    let layer1 = Bias {
        connections: radonn::nn::net::rand_range(5, 0.0f32, 0.6f32),
    };
    let layer2 = Activation {
        activation: Box::new(ReLU {}),
    };
    let layer3 = Dense {
        connections: radonn::nn::net::rand_range_xavier(5, 1),
        in_num: 5,
        out_num: 1,
    };
    let layer4 = Bias {
        connections: radonn::nn::net::rand_range(1, 0.0f32, 0.6f32),
    };
    let layer5 = Activation {
        activation: Box::new(Sigmoid {}),
    };

    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(layer0),
        Box::new(layer1),
        Box::new(layer2),
        Box::new(layer3),
        Box::new(layer4),
        Box::new(layer5),
    ];
    let mut nn = SeqNet { layers };

    let mut optim = Box::new(RMSProp::new(&nn, 1e-3f32, 0.999f32));

    let input = vec![vec![0.0, 0.0f32], vec![0.0, 1.0f32], vec![1.0, 0.0f32], vec![1.0, 1.0f32]];
    let output = vec![vec![0.0f32], vec![1.0f32], vec![1.0f32], vec![0.0f32]];

    let time = Instant::now();
    for _i in 0..100000 {
        for _j in 0..1 {
            let index = (thread_rng().gen::<f32>() * (input.len() as f32)) as usize;
            radonn::nn::optim::back_propagate(&input[index], &output[index], &mut nn, &mut optim);
        }
        radonn::nn::optim::step(&mut nn, &mut optim);
    }
    let duration = time.elapsed();
    println!("{}", duration.as_nanos());

    for index in 0..4 {
        let out = feed_forward(&input[index], &nn);
        print_vec(&input[index]);
        print_vec(&out);
        println!();
    }
}

pub fn print_vec<T: Display>(vector: &Vec<T>) {
    for f in vector {
        print!("{} ", f);
    }
}