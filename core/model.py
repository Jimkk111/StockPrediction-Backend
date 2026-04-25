import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from core.utils import Timer
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Model():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		# Keras 3.x 与旧版 .h5 不兼容（time_major 参数已删除，mse 短名也无法反序列化）
		# 统一走手动重建路径，直接跳过 keras.load_model()
		self._load_model_weights_compat(filepath)

	def _load_model_weights_compat(self, filepath):
		"""
		Keras 3.x 兼容加载旧版 .h5 文件：
		1. 读取 model_config，剔除 Keras 3 不支持的参数
		2. 按 config 手动重建 Sequential 网络
		3. 从 h5 加载权重
		"""
		import h5py
		import json

		# 不兼容的 LSTM/RNN 参数（Keras 3 已移除）
		_REMOVED_LSTM_KWARGS = {'time_major', 'implementation'}

		with h5py.File(filepath, 'r') as f:
			model_config = json.loads(f.attrs['model_config'])
			layers_cfg = model_config['config']['layers']

		# 重建 Sequential 模型
		from keras.models import Sequential
		from keras.layers import LSTM, Dropout, Dense, Activation

		model = Sequential()
		input_shape_set = False

		for layer_cfg in layers_cfg:
			cls_name = layer_cfg['class_name']
			cfg = dict(layer_cfg['config'])

			if cls_name == 'InputLayer':
				continue  # Sequential 会自动处理

			elif cls_name == 'LSTM':
				# 清理 Keras 3 不认的参数
				for bad_key in _REMOVED_LSTM_KWARGS:
					cfg.pop(bad_key, None)
				# 只保留 LSTM 支持的关键字
				lstm_kwargs = {
					'units': cfg['units'],
					'return_sequences': cfg.get('return_sequences', False),
					'activation': cfg.get('activation', 'tanh'),
					'recurrent_activation': cfg.get('recurrent_activation', 'sigmoid'),
					'use_bias': cfg.get('use_bias', True),
					'dropout': cfg.get('dropout', 0.0),
					'recurrent_dropout': cfg.get('recurrent_dropout', 0.0),
					'name': cfg['name'],
				}
				# 第一个 LSTM 层需要 input_shape
				if not input_shape_set:
					batch_shape = cfg.get('batch_input_shape')
					if batch_shape and len(batch_shape) == 3:
						lstm_kwargs['input_shape'] = (batch_shape[1], batch_shape[2])
						input_shape_set = True
				model.add(LSTM(**lstm_kwargs))

			elif cls_name == 'Dropout':
				model.add(Dropout(rate=cfg['rate'], name=cfg['name']))

			elif cls_name == 'Dense':
				dense_kwargs = {
					'units': cfg['units'],
					'activation': cfg.get('activation', 'linear'),
					'use_bias': cfg.get('use_bias', True),
					'name': cfg['name'],
				}
				model.add(Dense(**dense_kwargs))

			elif cls_name == 'Activation':
				model.add(Activation(cfg['activation'], name=cfg['name']))

		# 触发 build（给随机输入跑一次 predict 以初始化权重）
		import numpy as np
		# Keras 3 中 input_shape 通过 compute_output_spec 或直接从 config 读取
		# 直接用已知的 batch_input_shape（从 config 里已解析）
		first_lstm_cfg = next(
			l['config'] for l in layers_cfg if l['class_name'] == 'LSTM'
		)
		batch_shape = first_lstm_cfg.get('batch_input_shape', [None, 49, 2])
		timesteps = batch_shape[1] if batch_shape[1] else 49
		features = batch_shape[2] if batch_shape[2] else 2
		dummy = np.zeros((1, timesteps, features), dtype='float32')
		model.predict(dummy, verbose=0)

		# 从 h5 加载权重
		with h5py.File(filepath, 'r') as f:
			weight_group = f['model_weights']
			for layer in model.layers:
				layer_name = layer.name
				if layer_name in weight_group:
					g = weight_group[layer_name]
					# 递归找到权重数组
					weight_names = [n.decode('utf8') if isinstance(n, bytes) else n
									for n in g.attrs.get('weight_names', [])]
					if weight_names:
						weights = [g[wn] for wn in weight_names]
						layer.set_weights(weights)

		self.model = model
		print('[Model] Model loaded via manual weight loading (Keras 3 compat mode)')

	def build_model(self, configs):
		timer = Timer()
		timer.start()

		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))

		self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

		print('[Model] Model Compiled')
		timer.stop()

	def train(self, x, y, epochs, batch_size, save_dir):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]
		self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
		self.model.save(save_fname)

		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

	def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]
		self.model.fit_generator(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			workers=1
		)
		
		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

	def predict_point_by_point(self, data):
		#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
		print('[Model] Predicting Point-by-Point...')
		predicted = self.model.predict(data)
		predicted = np.reshape(predicted, (predicted.size,))
		return predicted

	def predict_sequences_multiple(self, data, window_size, prediction_len):
		#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
		print('[Model] Predicting Sequences Multiple...')
		prediction_seqs = []
		for i in range(int(len(data)/prediction_len)):
			curr_frame = data[i*prediction_len]
			predicted = []
			for j in range(prediction_len):
				predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
				curr_frame = curr_frame[1:]
				curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
			prediction_seqs.append(predicted)
		return prediction_seqs

	def predict_sequence_full(self, data, window_size):
		#Shift the window by 1 new prediction each time, re-run predictions on new window
		print('[Model] Predicting Sequences Full...')
		curr_frame = data[0]
		predicted = []
		for i in range(len(data)):
			predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
		return predicted
