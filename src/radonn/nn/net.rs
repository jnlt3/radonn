extern crate rand;

pub struct SeqNet {
	pub layers: Vec<Box<dyn Layer>>,
}

pub trait Layer {
	fn param_num(&self) -> usize;
	fn feed_forward(&self, input: &Vec<f32>) -> Vec<f32>;
	fn d_feed_forward(&self, input: &Vec<f32>) -> [Vec<f32>; 2];
	fn back_propagate(&self, out_err: &Vec<f32>, cache: &Vec<f32>) -> Vec<f32>;
	fn calc_err(&self, prev_err: &Vec<f32>, input: &Vec<f32>) -> Vec<f32>;
	fn step(&mut self, step: &Vec<f32>);
}

pub struct Dense {
	pub connections: Vec<f32>,
	pub in_num: usize,
	pub out_num: usize,
}

pub struct Bias {
	pub connections: Vec<f32>,
}

pub struct Activation {
	pub activation: Box<dyn Function>,
}

/*
pub struct Conv2D {
	pub filter: Vec<f32>,
	pub filter_width: usize,
	pub filter_height: usize,
	pub num_filters: usize,
	pub in_width: usize,
	pub in_height: usize,
	pub in_depth: usize,
	pub img_size: usize,
}

impl Conv2D {
	pub fn new(
		filter: Vec<f32>,
		filter_width: usize,
		filter_height: usize,
		num_filters: usize,
		in_width: usize,
		in_height: usize,
		in_depth: usize,
	) -> Self {
		Conv2D {
			filter,
			filter_width,
			filter_height,
			num_filters,
			in_width,
			in_height,
			in_depth,
			img_size: in_width * in_height,
		}
	}
}
 */

pub struct Softmax {}

impl Layer for Dense {
	fn param_num(&self) -> usize {
		return self.connections.len();
	}

	fn feed_forward(&self, input: &Vec<f32>) -> Vec<f32> {
		let mut out = vec!(0f32; self.out_num);
		for (con, in_v) in self.connections.chunks(self.out_num).zip(input) {
			for (out_ref, weight) in out.iter_mut().zip(con.iter()) {
				*out_ref += *weight * in_v;
			}
		}
		return out;
	}

	fn d_feed_forward(&self, input: &Vec<f32>) -> [Vec<f32>; 2] {
		return [self.feed_forward(&input), vec!()];
	}

	fn back_propagate(&self, out_err: &Vec<f32>, _cache: &Vec<f32>) -> Vec<f32> {
		let mut err = Vec::with_capacity(self.in_num);
		for con in self.connections.chunks(self.out_num) {
			let mut num = con[0] * out_err[0];
			for (weight, val) in con[1..self.out_num].iter().zip(out_err[1..out_err.len()].iter()) {
				num += weight * val;
			}
			err.push(num);
		}
		return err;
	}

	fn calc_err(&self, prev_err: &Vec<f32>, input: &Vec<f32>) -> Vec<f32> {
		let mut connect_errors = Vec::with_capacity(self.connections.len());
		for err in prev_err {
			for in_val in input {
				connect_errors.push(err * in_val);
			}
		}
		return connect_errors;
	}

	fn step(&mut self, step: &Vec<f32>) {
		for (val, step) in self.connections.iter_mut().zip(step) {
			*val -= step;
		}
	}
}

impl Layer for Bias {
	fn param_num(&self) -> usize {
		return self.connections.len();
	}

	fn feed_forward(&self, input: &Vec<f32>) -> Vec<f32> {
		let inputs = self.connections.len();
		let mut out = Vec::with_capacity(inputs);
		for i in 0..input.len() {
			out.push(input[i] + self.connections[i]);
		}
		return out;
	}

	fn d_feed_forward(&self, input: &Vec<f32>) -> [Vec<f32>; 2] {
		return [self.feed_forward(&input), vec!()];
	}

	fn back_propagate(&self, out_err: &Vec<f32>, _cache: &Vec<f32>) -> Vec<f32> {
		return out_err.clone();
	}

	fn calc_err(&self, prev_err: &Vec<f32>, _input: &Vec<f32>) -> Vec<f32> {
		return prev_err.clone();
	}

	fn step(&mut self, step: &Vec<f32>) {
		for (val, step) in self.connections.iter_mut().zip(step) {
			*val -= step;
		}
	}
}

impl Layer for Activation {
	fn param_num(&self) -> usize {
		return 0usize;
	}

	fn feed_forward(&self, input: &Vec<f32>) -> Vec<f32> {
		let ret_val = self.activation.activate(&input);
		return ret_val;
	}

	fn d_feed_forward(&self, input: &Vec<f32>) -> [Vec<f32>; 2] {
		return self.activation.d_activate(&input);
	}

	fn back_propagate(&self, out: &Vec<f32>, cache: &Vec<f32>) -> Vec<f32> {
		let mut delta = Vec::with_capacity(out.len());
		for i in 0..out.len() {
			delta.push(out[i] * cache[i]);
		}
		return delta;
	}

	fn calc_err(&self, _prev_err: &Vec<f32>, _input: &Vec<f32>) -> Vec<f32> {
		return vec!();
	}

	fn step(&mut self, _step: &Vec<f32>) {}
}

/*
impl Layer for Conv2D {
	fn param_num(&self) -> usize {
		return self.filter.len();
	}

	fn feed_forward(&self, input: &Vec<f32>) -> Vec<f32> {
		for img in input.chunks(self.img_size) {

		}
		()
	}

	fn d_feed_forward(&self, input: &Vec<f32>) -> [Vec<f32>; 2] {
		unimplemented!()
	}

	fn back_propagate(&self, out_err: &Vec<f32>, cache: &Vec<f32>) -> Vec<f32> {
		unimplemented!()
	}

	fn calc_err(&self, prev_err: &Vec<f32>, input: &Vec<f32>) -> Vec<f32> {
		unimplemented!()
	}

	fn step(&mut self, step: &Vec<f32>) {
		for (val, step) in self.connections.iter_mut().zip(step) {
			*val -= step;
		}
	}
}
 */

impl Layer for Softmax {
	fn param_num(&self) -> usize {
		return 0usize;
	}

	fn feed_forward(&self, input: &Vec<f32>) -> Vec<f32> {
		let mut max = input[0];
		for i in 0..input.len() {
			if input[i] > max {
				max = input[i];
			}
		}
		let mut exps = Vec::with_capacity(input.len());
		let mut sum = 0f32;
		for i in 0..input.len() {
			let val = (input[i] - max).exp();
			sum += val;
			exps.push(val);
		}
		for i in 0..exps.len() {
			exps[i] /= sum;
		}
		return exps;
	}

	fn d_feed_forward(&self, input: &Vec<f32>) -> [Vec<f32>; 2] {
		return [self.feed_forward(input), vec!()];
	}

	fn back_propagate(&self, out_err: &Vec<f32>, _cache: &Vec<f32>) -> Vec<f32> {
		return out_err.clone();
	}

	fn calc_err(&self, _prev_err: &Vec<f32>, _input: &Vec<f32>) -> Vec<f32> {
		return vec!();
	}

	fn step(&mut self, _step: &Vec<f32>) {}
}

pub fn rand_range_xavier(inputs: usize, outputs: usize) -> Vec<f32> {
	let range = 6f32.sqrt() / ((inputs + outputs) as f32);
	return rand_range(inputs * outputs, -range, range);
}

pub fn rand_range_conv2d(width: usize, height: usize, depth: usize) -> Vec<f32> {
	let num = (width * height * depth);
	let max = (2f32 / (num as f32)).sqrt();
	return rand_range(num, 0f32, max);
}

pub fn rand_range(len: usize, min: f32, max: f32) -> Vec<f32> {
	let diff = max - min;
	let mut connections = Vec::new();
	for _i in 0..len {
		connections.push(rand::random::<f32>() * diff + min);
	}
	return connections;
}

pub struct ReLU {}

pub struct Sigmoid {}


pub trait Function {
	fn activate(&self, val: &Vec<f32>) -> Vec<f32>;

	fn d_activate(&self, val: &Vec<f32>) -> [Vec<f32>; 2];
}

impl Function for ReLU {
	fn activate(&self, val: &Vec<f32>) -> Vec<f32> {
		return val.into_iter().map(|&val| if val > 0f32 { val } else { 0f32 }).collect();
	}

	fn d_activate(&self, val: &Vec<f32>) -> [Vec<f32>; 2] {
		let res = val.into_iter().map(|&val| if val > 0f32 { (val, 1f32) } else { (0f32, 0f32) }).unzip();
		return [res.0, res.1];
	}
}

impl Function for Sigmoid {
	fn activate(&self, val: &Vec<f32>) -> Vec<f32> {
		return val.into_iter().map(|&val| 1f32 / (1f32 + (-val).exp())).collect();
	}

	fn d_activate(&self, val: &Vec<f32>) -> [Vec<f32>; 2] {
		let res = val.into_iter().map(|&val| sigmoid_w_derivation(val)).unzip();
		return [res.0, res.1];
	}
}

fn sigmoid_w_derivation(val: f32) -> (f32, f32) {
	let sigmoid = 1f32 / (1f32 + (-val).exp());
	return (sigmoid, sigmoid * (1f32 - sigmoid));
}