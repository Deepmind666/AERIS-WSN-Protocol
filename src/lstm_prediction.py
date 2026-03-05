# 鍩轰簬LSTM鐨勬繁搴﹀涔犻娴嬫ā鍧?- 鐢ㄤ簬AERIS鍗忚
# 鍙傝€冩枃鐚細
# [8] Alsheikh, M. A., Lin, S., Niyato, D., & Tan, H. P. (2014). Machine learning in wireless sensor networks: Algorithms, strategies, and applications.
# [9] Raza, U., Kulkarni, P., & Sooriyabandara, M. (2017). Low power wide area networks: An overview.
# [10] Fadel, E., Gungor, V. C., Nassef, L., Akkari, N., Malik, M. A., Almasri, S., & Akyildiz, I. F. (2015). A survey on wireless sensor networks for smart grid.
# [11] Luo, C., Wu, F., Sun, J., & Chen, C. W. (2015). Compressive data gathering for large-scale wireless sensor networks.
# [12] Wang, J., Chen, Y., Hao, S., Peng, X., & Hu, L. (2019). Deep learning for sensor-based activity recognition: A survey.
# [13] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
# [14] Mao, B., Kawamoto, Y., & Kato, N. (2020). AI-Based Joint Optimization of QoS and Security for 6G Energy Harvesting Internet of Things.
# [15] Qiu, T., Chen, N., Li, K., Atiquzzaman, M., & Zhao, W. (2018). How can heterogeneous Internet of Things build our future: A survey.

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class LSTMPrediction:
    """鍩轰簬LSTM鐨勬繁搴﹀涔犻娴嬫ā鍧楋紝鐢ㄤ簬WSN涓殑缃戠粶娴侀噺銆佽妭鐐规晠闅滃拰閾捐矾璐ㄩ噺棰勬祴
    
    鍙傝€冩枃鐚細
    [12] Wang, J., Chen, Y., Hao, S., Peng, X., & Hu, L. (2019). Deep learning in sensor-based activity recognition: A survey.
    [13] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
    [14] Mao, B., Kawamoto, Y., & Kato, N. (2020). AI-Based Joint Optimization of QoS and Security for 6G Energy Harvesting Internet of Things.
    """
    
    def __init__(self, sequence_length=10, prediction_horizon=5, multi_feature=False):
        """鍒濆鍖朙STM棰勬祴妯″瀷
        
        鍙傛暟:
            sequence_length: 鐢ㄤ簬棰勬祴鐨勫巻鍙叉暟鎹暱搴?            prediction_horizon: 棰勬祴鐨勬湭鏉ユ椂闂存闀垮害
            multi_feature: 鏄惁浣跨敤澶氱壒寰佽緭鍏ワ紙榛樿涓篎alse锛屼粎浣跨敤鍗曠壒寰侊級
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.multi_feature = multi_feature
        
        # 棰勬祴妯″瀷
        self.traffic_model = None
        self.energy_model = None
        self.link_quality_model = None
        
        # 鏁版嵁鏍囧噯鍖栧櫒
        self.traffic_scaler = MinMaxScaler(feature_range=(0, 1))
        self.energy_scaler = MinMaxScaler(feature_range=(0, 1))
        self.link_quality_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # 妯″瀷璁粌鐘舵€?        self.is_trained_traffic = False
        self.is_trained_energy = False
        self.is_trained_link_quality = False
        
        # 妯″瀷鐗堟湰鍜屾洿鏂版椂闂?        self.model_version = 1.0
        self.last_update = None
        
    def build_model(self, input_shape, output_dim=1):
        """鏋勫缓LSTM妯″瀷
        
        鍙傛暟:
            input_shape: 杈撳叆鏁版嵁褰㈢姸 (sequence_length, features)
            output_dim: 杈撳嚭缁村害
            
        杩斿洖:
            鏋勫缓濂界殑LSTM妯″瀷
        """
        model = Sequential()
        
        # 绗竴灞侺STM锛岃繑鍥炲簭鍒?        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        # 绗簩灞侺STM锛屼笉杩斿洖搴忓垪
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        
        # 杈撳嚭灞?        model.add(Dense(output_dim))
        
        # 缂栬瘧妯″瀷
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def prepare_data(self, data, is_traffic=True, is_link_quality=False):
        """鍑嗗LSTM妯″瀷鐨勫崟鍙橀噺璁粌鏁版嵁
        
        鍙傛暟:
            data: 鍘熷鏃堕棿搴忓垪鏁版嵁
            is_traffic: 鏄惁涓烘祦閲忔暟鎹?            is_link_quality: 鏄惁涓洪摼璺川閲忔暟鎹?            
        杩斿洖:
            X: 杈撳叆搴忓垪
            y: 鐩爣鍊?        """
        # 閫夋嫨鍚堥€傜殑鏍囧噯鍖栧櫒
        if is_traffic:
            scaler = self.traffic_scaler
        elif is_link_quality:
            scaler = self.link_quality_scaler
        else:
            scaler = self.energy_scaler
            
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        
        # 鍒涘缓杈撳叆搴忓垪鍜岀洰鏍囧€?        for i in range(len(scaled_data) - self.sequence_length - self.prediction_horizon + 1):
            X.append(scaled_data[i:(i + self.sequence_length), 0])
            y.append(scaled_data[i + self.sequence_length:(i + self.sequence_length + self.prediction_horizon), 0])
        
        # 杞崲涓簄umpy鏁扮粍
        X = np.array(X)
        y = np.array(y)
        
        # 閲嶅涓篖STM杈撳叆鏍煎紡 [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
        
    def prepare_multivariate_data(self, data_dict, target_type='traffic'):
        """鍑嗗LSTM妯″瀷鐨勫鍙橀噺璁粌鏁版嵁
        
        鍙傛暟:
            data_dict: 鍖呭惈澶氫釜鐗瑰緛鐨勫瓧鍏革紝鏍煎紡涓?{feature_name: feature_data}
            target_type: 鐩爣鍙橀噺绫诲瀷锛屽彲閫夊€间负 'traffic', 'energy', 'link_quality'
            
        杩斿洖:
            X: 杈撳叆搴忓垪
            y: 鐩爣鍊?        """
        # 纭畾鐩爣鍙橀噺鍜屽搴旂殑鏍囧噯鍖栧櫒
        if target_type == 'traffic':
            target_key = next((k for k in data_dict.keys() if 'traffic' in k.lower()), list(data_dict.keys())[0])
            target_scaler = self.traffic_scaler
        elif target_type == 'link_quality':
            target_key = next((k for k in data_dict.keys() if 'link' in k.lower() or 'quality' in k.lower()), list(data_dict.keys())[0])
            target_scaler = self.link_quality_scaler
        else:  # energy
            target_key = next((k for k in data_dict.keys() if 'energy' in k.lower()), list(data_dict.keys())[0])
            target_scaler = self.energy_scaler
        
        # 鎻愬彇鐩爣鍙橀噺
        target_data = data_dict[target_key]
        
        # 鏍囧噯鍖栫洰鏍囧彉閲?        scaled_target = target_scaler.fit_transform(target_data.reshape(-1, 1))
        
        # 鏍囧噯鍖栨墍鏈夌壒寰?        scaled_features = {}
        feature_scalers = {}
        
        for feature_name, feature_data in data_dict.items():
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features[feature_name] = scaler.fit_transform(feature_data.reshape(-1, 1))
            feature_scalers[feature_name] = scaler
        
        # 鍒涘缓澶氬彉閲忚緭鍏ュ簭鍒楀拰鐩爣鍊?        X, y = [], []
        
        for i in range(len(scaled_target) - self.sequence_length - self.prediction_horizon + 1):
            # 鏀堕泦鎵€鏈夌壒寰佺殑搴忓垪
            features_sequence = []
            for feature_name in data_dict.keys():
                features_sequence.append(scaled_features[feature_name][i:(i + self.sequence_length), 0])
            
            # 灏嗘墍鏈夌壒寰佸爢鍙犱负涓€涓鍙橀噺搴忓垪
            X.append(np.column_stack(features_sequence))
            
            # 鐩爣鍊间粛鐒舵槸鍗曞彉閲?            y.append(scaled_target[i + self.sequence_length:(i + self.sequence_length + self.prediction_horizon), 0])
        
        # 杞崲涓簄umpy鏁扮粍
        X = np.array(X)
        y = np.array(y)
        
        return X, y, target_scaler, feature_scalers
    
    def train_traffic_model(self, traffic_data, epochs=50, batch_size=32, validation_split=0.2, multi_features=None):
        """璁粌缃戠粶娴侀噺棰勬祴妯″瀷
        
        鍙傛暟:
            traffic_data: 缃戠粶娴侀噺鍘嗗彶鏁版嵁
            epochs: 璁粌杞暟
            batch_size: 鎵规澶у皬
            validation_split: 楠岃瘉闆嗘瘮渚?            multi_features: 澶氱壒寰佽緭鍏ュ瓧鍏革紝鏍煎紡涓?{feature_name: feature_data}
            
        杩斿洖:
            璁粌鍘嗗彶
        """
        print("\n[TRAIN] Start training traffic prediction model...")
        print(f"[CONFIG] epochs={epochs}, batch_size={batch_size}, validation_split={validation_split}")
        
        if multi_features is not None and self.multi_feature:
            # 浣跨敤澶氱壒寰佽緭鍏?            data_dict = multi_features.copy()
            data_dict['traffic'] = traffic_data
            X, y, target_scaler, _ = self.prepare_multivariate_data(data_dict, target_type='traffic')
            input_shape = (self.sequence_length, len(data_dict))
            print(f"[INPUT] Use multi-feature inputs: {list(data_dict.keys())}")
        else:
            # 浣跨敤鍗曠壒寰佽緭鍏?            X, y = self.prepare_data(traffic_data, is_traffic=True)
            input_shape = (self.sequence_length, 1)
            print("[INPUT] Use single-feature input: traffic series")
        
        print(f"[SHAPE] X={X.shape}, y={y.shape}")
        
        # 鏋勫缓妯″瀷
        self.traffic_model = self.build_model(input_shape, self.prediction_horizon)
        
        # 鏃╁仠鍥炶皟
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # 杩涘害鎵撳嵃鍥炶皟
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 5 == 0 or epoch == epochs - 1:  # 姣?涓猠poch鎵撳嵃涓€娆★紝浠ュ強鏈€鍚庝竴涓猠poch
                    progress = (epoch + 1) / epochs * 100
                    print(f"[PROGRESS] {progress:.1f}% (Epoch {epoch+1}/{epochs}), loss: {logs['loss']:.4f}, val_loss: {logs['val_loss']:.4f}")
        
        # 璁粌妯″瀷
        history = self.traffic_model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, ProgressCallback()],
            verbose=0  # 鍏抽棴榛樿杩涘害鏉★紝浣跨敤鑷畾涔夎繘搴︽樉绀?        )
        
        self.is_trained_traffic = True
        print("[OK] Traffic model training completed!")
        return history
    
    def train_energy_model(self, energy_data, epochs=50, batch_size=32, validation_split=0.2, multi_features=None):
        """璁粌鑺傜偣鑳介噺娑堣€楅娴嬫ā鍨?        
        鍙傛暟:
            energy_data: 鑺傜偣鑳介噺鍘嗗彶鏁版嵁
            epochs: 璁粌杞暟
            batch_size: 鎵规澶у皬
            validation_split: 楠岃瘉闆嗘瘮渚?            multi_features: 澶氱壒寰佽緭鍏ュ瓧鍏革紝鏍煎紡涓?{feature_name: feature_data}
            
        杩斿洖:
            璁粌鍘嗗彶
        """
        if multi_features is not None and self.multi_feature:
            # 浣跨敤澶氱壒寰佽緭鍏?            data_dict = multi_features.copy()
            data_dict['energy'] = energy_data
            X, y, target_scaler, _ = self.prepare_multivariate_data(data_dict, target_type='energy')
            input_shape = (self.sequence_length, len(data_dict))
        else:
            # 浣跨敤鍗曠壒寰佽緭鍏?            X, y = self.prepare_data(energy_data, is_traffic=False)
            input_shape = (self.sequence_length, 1)
        
        # 鏋勫缓妯″瀷
        self.energy_model = self.build_model(input_shape, self.prediction_horizon)
        
        # 鏃╁仠鍥炶皟
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # 璁粌妯″瀷
        history = self.energy_model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.is_trained_energy = True
        return history
        
    def train_link_quality_model(self, link_quality_data, epochs=50, batch_size=32, validation_split=0.2, multi_features=None):
        """璁粌閾捐矾璐ㄩ噺棰勬祴妯″瀷
        
        鍙傛暟:
            link_quality_data: 閾捐矾璐ㄩ噺鍘嗗彶鏁版嵁
            epochs: 璁粌杞暟
            batch_size: 鎵规澶у皬
            validation_split: 楠岃瘉闆嗘瘮渚?            multi_features: 澶氱壒寰佽緭鍏ュ瓧鍏革紝鏍煎紡涓?{feature_name: feature_data}
            
        杩斿洖:
            璁粌鍘嗗彶
        """
        if multi_features is not None and self.multi_feature:
            # 浣跨敤澶氱壒寰佽緭鍏?            data_dict = multi_features.copy()
            data_dict['link_quality'] = link_quality_data
            X, y, target_scaler, _ = self.prepare_multivariate_data(data_dict, target_type='link_quality')
            input_shape = (self.sequence_length, len(data_dict))
        else:
            # 浣跨敤鍗曠壒寰佽緭鍏?            X, y = self.prepare_data(link_quality_data, is_traffic=False, is_link_quality=True)
            input_shape = (self.sequence_length, 1)
        
        # 鏋勫缓妯″瀷
        self.link_quality_model = self.build_model(input_shape, self.prediction_horizon)
        
        # 鏃╁仠鍥炶皟
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # 璁粌妯″瀷
        history = self.link_quality_model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.is_trained_link_quality = True
        return history
    
    def predict_traffic(self, recent_traffic, multi_features=None):
        """棰勬祴鏈潵缃戠粶娴侀噺
        
        鍙傛暟:
            recent_traffic: 鏈€杩戠殑娴侀噺鏁版嵁锛岄暱搴﹀簲涓簊equence_length
            multi_features: 澶氱壒寰佽緭鍏ュ瓧鍏革紝鏍煎紡涓?{feature_name: feature_data}
            
        杩斿洖:
            棰勬祴鐨勬湭鏉ユ祦閲忔暟鎹?        """
        if not self.is_trained_traffic:
            raise ValueError("娴侀噺棰勬祴妯″瀷灏氭湭璁粌")
        
        # 纭繚杈撳叆闀垮害姝ｇ‘
        if len(recent_traffic) != self.sequence_length:
            raise ValueError(f"杈撳叆鏁版嵁闀垮害搴斾负{self.sequence_length}")
        
        if multi_features is not None and self.multi_feature:
            # 浣跨敤澶氱壒寰佽緭鍏?            data_dict = multi_features.copy()
            data_dict['traffic'] = recent_traffic
            
            # 鏍囧噯鍖栨墍鏈夌壒寰?            scaled_features = {}
            for feature_name, feature_data in data_dict.items():
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_features[feature_name] = scaler.fit_transform(feature_data.reshape(-1, 1))
            
            # 鏀堕泦鎵€鏈夌壒寰佺殑搴忓垪
            features_sequence = []
            for feature_name in data_dict.keys():
                features_sequence.append(scaled_features[feature_name][:self.sequence_length, 0])
            
            # 灏嗘墍鏈夌壒寰佸爢鍙犱负涓€涓鍙橀噺搴忓垪
            X = np.column_stack(features_sequence)
            X = np.reshape(X, (1, X.shape[0], X.shape[1]))
        else:
            # 浣跨敤鍗曠壒寰佽緭鍏?            # 鏁版嵁鏍囧噯鍖?            scaled_data = self.traffic_scaler.transform(recent_traffic.reshape(-1, 1))
            
            # 閲嶅涓篖STM杈撳叆鏍煎紡
            X = np.reshape(scaled_data, (1, self.sequence_length, 1))
        
        # 棰勬祴
        scaled_prediction = self.traffic_model.predict(X)
        
        # 鍙嶆爣鍑嗗寲
        prediction = self.traffic_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))
        
        return prediction.flatten()
    
    def predict_energy(self, recent_energy, multi_features=None):
        """棰勬祴鏈潵鑺傜偣鑳介噺娑堣€?        
        鍙傛暟:
            recent_energy: 鏈€杩戠殑鑳介噺鏁版嵁锛岄暱搴﹀簲涓簊equence_length
            multi_features: 澶氱壒寰佽緭鍏ュ瓧鍏革紝鏍煎紡涓?{feature_name: feature_data}
            
        杩斿洖:
            棰勬祴鐨勬湭鏉ヨ兘閲忔暟鎹?        """
        if not self.is_trained_energy:
            raise ValueError("鑳介噺棰勬祴妯″瀷灏氭湭璁粌")
        
        # 纭繚杈撳叆闀垮害姝ｇ‘
        if len(recent_energy) != self.sequence_length:
            raise ValueError(f"杈撳叆鏁版嵁闀垮害搴斾负{self.sequence_length}")
        
        if multi_features is not None and self.multi_feature:
            # 浣跨敤澶氱壒寰佽緭鍏?            data_dict = multi_features.copy()
            data_dict['energy'] = recent_energy
            
            # 鏍囧噯鍖栨墍鏈夌壒寰?            scaled_features = {}
            for feature_name, feature_data in data_dict.items():
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_features[feature_name] = scaler.fit_transform(feature_data.reshape(-1, 1))
            
            # 鏀堕泦鎵€鏈夌壒寰佺殑搴忓垪
            features_sequence = []
            for feature_name in data_dict.keys():
                features_sequence.append(scaled_features[feature_name][:self.sequence_length, 0])
            
            # 灏嗘墍鏈夌壒寰佸爢鍙犱负涓€涓鍙橀噺搴忓垪
            X = np.column_stack(features_sequence)
            X = np.reshape(X, (1, X.shape[0], X.shape[1]))
        else:
            # 浣跨敤鍗曠壒寰佽緭鍏?            # 鏁版嵁鏍囧噯鍖?            scaled_data = self.energy_scaler.transform(recent_energy.reshape(-1, 1))
            
            # 閲嶅涓篖STM杈撳叆鏍煎紡
            X = np.reshape(scaled_data, (1, self.sequence_length, 1))
        
        # 棰勬祴
        scaled_prediction = self.energy_model.predict(X)
        
        # 鍙嶆爣鍑嗗寲
        prediction = self.energy_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))
        
        return prediction.flatten()
    
    def predict_link_quality(self, recent_link_quality, multi_features=None):
        """棰勬祴鏈潵閾捐矾璐ㄩ噺
        
        鍙傛暟:
            recent_link_quality: 鏈€杩戠殑閾捐矾璐ㄩ噺鏁版嵁锛岄暱搴﹀簲涓簊equence_length
            multi_features: 澶氱壒寰佽緭鍏ュ瓧鍏革紝鏍煎紡涓?{feature_name: feature_data}
            
        杩斿洖:
            棰勬祴鐨勬湭鏉ラ摼璺川閲忔暟鎹?        """
        if not self.is_trained_link_quality:
            raise ValueError("閾捐矾璐ㄩ噺棰勬祴妯″瀷灏氭湭璁粌")
        
        # 纭繚杈撳叆闀垮害姝ｇ‘
        if len(recent_link_quality) != self.sequence_length:
            raise ValueError(f"杈撳叆鏁版嵁闀垮害搴斾负{self.sequence_length}")
        
        if multi_features is not None and self.multi_feature:
            # 浣跨敤澶氱壒寰佽緭鍏?            data_dict = multi_features.copy()
            data_dict['link_quality'] = recent_link_quality
            
            # 鏍囧噯鍖栨墍鏈夌壒寰?            scaled_features = {}
            for feature_name, feature_data in data_dict.items():
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_features[feature_name] = scaler.fit_transform(feature_data.reshape(-1, 1))
            
            # 鏀堕泦鎵€鏈夌壒寰佺殑搴忓垪
            features_sequence = []
            for feature_name in data_dict.keys():
                features_sequence.append(scaled_features[feature_name][:self.sequence_length, 0])
            
            # 灏嗘墍鏈夌壒寰佸爢鍙犱负涓€涓鍙橀噺搴忓垪
            X = np.column_stack(features_sequence)
            X = np.reshape(X, (1, X.shape[0], X.shape[1]))
        else:
            # 浣跨敤鍗曠壒寰佽緭鍏?            # 鏁版嵁鏍囧噯鍖?            scaled_data = self.link_quality_scaler.transform(recent_link_quality.reshape(-1, 1))
            
            # 閲嶅涓篖STM杈撳叆鏍煎紡
            X = np.reshape(scaled_data, (1, self.sequence_length, 1))
        
        # 棰勬祴
        scaled_prediction = self.link_quality_model.predict(X)
        
        # 鍙嶆爣鍑嗗寲
        prediction = self.link_quality_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))
        
        return prediction.flatten()
    
    def predict_node_failure(self, recent_energy, energy_threshold=0.1, time_steps=None, multi_features=None, additional_factors=None):
        """棰勬祴鑺傜偣鏁呴殰
        
        鍙傛暟:
            recent_energy: 鏈€杩戠殑鑳介噺鏁版嵁
            energy_threshold: 鑳介噺闃堝€硷紝浣庝簬姝ゅ€艰涓烘晠闅?            time_steps: 棰勬祴鐨勬椂闂存鏁帮紝榛樿涓簆rediction_horizon
            multi_features: 澶氱壒寰佽緭鍏ュ瓧鍏革紝鐢ㄤ簬鑳介噺棰勬祴
            additional_factors: 褰卞搷鑺傜偣鏁呴殰鐨勯澶栧洜绱犲瓧鍏革紝鏍煎紡涓?{factor_name: weight}
            
        杩斿洖:
            (failure_predicted, time_to_failure, failure_probability): 鏄惁棰勬祴鍒版晠闅滐紝棰勮鏁呴殰鏃堕棿锛屼互鍙婃晠闅滄鐜?        """
        if time_steps is None:
            time_steps = self.prediction_horizon
        
        # 棰勬祴鏈潵鑳介噺
        if multi_features is not None and self.multi_feature:
            # 浣跨敤澶氱壒寰侀娴嬭兘閲?            future_energy = self.predict_energy(recent_energy, multi_features)
        else:
            future_energy = self.predict_energy(recent_energy)
        
        # 鍩虹鏁呴殰妫€娴嬶細妫€鏌ユ槸鍚︽湁棰勬祴鍊间綆浜庨槇鍊?        failure_indices = np.where(future_energy < energy_threshold)[0]
        
        # 璁＄畻鍩虹鏁呴殰姒傜巼
        if len(failure_indices) > 0:
            # 鎵惧埌绗竴娆℃晠闅滅殑鏃堕棿姝?            time_to_failure = failure_indices[0] + 1  # +1鍥犱负绱㈠紩浠?寮€濮?            base_failure_probability = 1.0 - (future_energy[failure_indices[0]] / energy_threshold)
        else:
            time_to_failure = None
            min_energy = np.min(future_energy)
            base_failure_probability = 0.5 * (1.0 - (min_energy / energy_threshold))
            if base_failure_probability < 0:
                base_failure_probability = 0
        
        # 鑰冭檻棰濆鍥犵礌
        final_failure_probability = base_failure_probability
        if additional_factors is not None:
            # 棰濆鍥犵礌鏉冮噸鎬诲拰
            total_weight = sum(additional_factors.values())
            
            # 璁＄畻棰濆鍥犵礌鐨勫奖鍝?            for factor, weight in additional_factors.items():
                if factor == 'traffic_congestion' and 'congestion_probability' in additional_factors:
                    # 缃戠粶鎷ュ浼氬鍔犳晠闅滄鐜?                    final_failure_probability += (additional_factors['congestion_probability'] * weight / total_weight)
                elif factor == 'link_quality' and 'link_quality_value' in additional_factors:
                    # 閾捐矾璐ㄩ噺宸細澧炲姞鏁呴殰姒傜巼
                    link_quality = additional_factors['link_quality_value']
                    link_quality_factor = 1.0 - link_quality  # 閾捐矾璐ㄩ噺瓒婁綆锛屽奖鍝嶈秺澶?                    final_failure_probability += (link_quality_factor * weight / total_weight)
                elif factor == 'temperature' and 'temperature_value' in additional_factors:
                    # 娓╁害杩囬珮浼氬鍔犳晠闅滄鐜?                    temp = additional_factors['temperature_value']
                    temp_threshold = additional_factors.get('temperature_threshold', 70)  # 榛樿娓╁害闃堝€?                    if temp > temp_threshold:
                        temp_factor = (temp - temp_threshold) / 30  # 鍋囪娓╁害瓒呰繃闃堝€?0搴︿細瀵艰嚧100%鏁呴殰
                        final_failure_probability += (min(temp_factor, 1.0) * weight / total_weight)
            
            # 纭繚姒傜巼鍦╗0,1]鑼冨洿鍐?            final_failure_probability = max(0.0, min(1.0, final_failure_probability))
        
        # 鏍规嵁鏈€缁堟鐜囩‘瀹氭槸鍚﹂娴嬪埌鎷ュ
        failure_predicted = final_failure_probability > 0.5 or (time_to_failure is not None)
        
        return failure_predicted, time_to_failure, final_failure_probability
    
    def predict_congestion(self, recent_traffic, capacity_threshold, time_steps=None, multi_features=None, additional_factors=None):
        """棰勬祴缃戠粶鎷ュ
        
        鍙傛暟:
            recent_traffic: 鏈€杩戠殑娴侀噺鏁版嵁
            capacity_threshold: 瀹归噺闃堝€硷紝楂樹簬姝ゅ€艰涓烘嫢濉?            time_steps: 棰勬祴鐨勬椂闂存鏁帮紝榛樿涓簆rediction_horizon
            multi_features: 澶氱壒寰佽緭鍏ュ瓧鍏革紝鐢ㄤ簬娴侀噺棰勬祴
            additional_factors: 褰卞搷鎷ュ鐨勯澶栧洜绱犲瓧鍏革紝鏍煎紡涓?{factor_name: weight}
            
        杩斿洖:
            (congestion_predicted, time_to_congestion, congestion_probability): 鏄惁棰勬祴鍒版嫢濉烇紝棰勮鎷ュ鏃堕棿锛屼互鍙婃嫢濉炴鐜?        """
        if time_steps is None:
            time_steps = self.prediction_horizon
        
        # 棰勬祴鏈潵娴侀噺
        if multi_features is not None and self.multi_feature:
            # 浣跨敤澶氱壒寰侀娴嬫祦閲?            future_traffic = self.predict_traffic(recent_traffic, multi_features)
        else:
            future_traffic = self.predict_traffic(recent_traffic)
        
        # 鍩虹鎷ュ妫€娴嬶細妫€鏌ユ槸鍚︽湁棰勬祴鍊奸珮浜庨槇鍊?        congestion_indices = np.where(future_traffic > capacity_threshold)[0]
        
        # 璁＄畻鍩虹鎷ュ姒傜巼
        if len(congestion_indices) > 0:
            # 鎵惧埌绗竴娆℃嫢濉炵殑鏃堕棿姝?            time_to_congestion = congestion_indices[0] + 1  # +1鍥犱负绱㈠紩浠?寮€濮?            base_congestion_probability = (future_traffic[congestion_indices[0]] / capacity_threshold) - 1.0
            base_congestion_probability = min(base_congestion_probability, 1.0)  # 纭繚姒傜巼涓嶈秴杩?
        else:
            time_to_congestion = None
            max_traffic = np.max(future_traffic)
            base_congestion_probability = 0.5 * (max_traffic / capacity_threshold)
            if base_congestion_probability > 1.0:
                base_congestion_probability = 1.0
        
        # 鑰冭檻棰濆鍥犵礌
        final_congestion_probability = base_congestion_probability
        if additional_factors is not None:
            # 棰濆鍥犵礌鏉冮噸鎬诲拰
            total_weight = sum(additional_factors.values())
            
            # 璁＄畻棰濆鍥犵礌鐨勫奖鍝?            for factor, weight in additional_factors.items():
                if factor == 'node_failures' and 'failure_count' in additional_factors:
                    # 鑺傜偣鏁呴殰浼氬鍔犳嫢濉炴鐜?                    failure_count = additional_factors['failure_count']
                    failure_factor = min(failure_count / 10.0, 1.0)  # 鍋囪10涓妭鐐规晠闅滀細瀵艰嚧100%鎷ュ
                    final_congestion_probability += (failure_factor * weight / total_weight)
                elif factor == 'link_quality' and 'link_quality_value' in additional_factors:
                    # 閾捐矾璐ㄩ噺宸細澧炲姞鎷ュ姒傜巼
                    link_quality = additional_factors['link_quality_value']
                    link_quality_factor = 1.0 - link_quality  # 閾捐矾璐ㄩ噺瓒婁綆锛屽奖鍝嶈秺澶?                    final_congestion_probability += (link_quality_factor * weight / total_weight)
                elif factor == 'packet_loss' and 'packet_loss_rate' in additional_factors:
                    # 涓㈠寘鐜囬珮浼氬鍔犳嫢濉炴鐜?                    packet_loss = additional_factors['packet_loss_rate']
                    final_congestion_probability += (packet_loss * weight / total_weight)
            
            # 纭繚姒傜巼鍦╗0,1]鑼冨洿鍐?            final_congestion_probability = max(0.0, min(1.0, final_congestion_probability))
        
        # 鏍规嵁鏈€缁堟鐜囩‘瀹氭槸鍚﹂娴嬪埌鎷ュ
        congestion_predicted = final_congestion_probability > 0.5 or (time_to_congestion is not None)
        
        return congestion_predicted, time_to_congestion, final_congestion_probability
    
    def visualize_prediction(self, actual_data, recent_data, prediction, title="棰勬祴缁撴灉"):
        """鍙鍖栭娴嬬粨鏋?        
        鍙傛暟:
            actual_data: 瀹為檯鏁版嵁锛堝鏋滄湁锛?            recent_data: 鏈€杩戠殑鍘嗗彶鏁版嵁
            prediction: 棰勬祴鏁版嵁
            title: 鍥捐〃鏍囬
        """
        plt.figure(figsize=(12, 6))
        
        # 缁樺埗鍘嗗彶鏁版嵁
        time_points = list(range(len(recent_data)))
        plt.plot(time_points, recent_data, 'b-', label='鍘嗗彶鏁版嵁')
        
        # 缁樺埗棰勬祴鏁版嵁
        future_time_points = list(range(len(recent_data), len(recent_data) + len(prediction)))
        plt.plot(future_time_points, prediction, 'r--', label='棰勬祴鏁版嵁')
        
        # 濡傛灉鏈夊疄闄呮湭鏉ユ暟鎹紝涔熺粯鍒跺嚭鏉?        if actual_data is not None and len(actual_data) >= len(prediction):
            plt.plot(future_time_points, actual_data[:len(prediction)], 'g-', label='瀹為檯鏁版嵁')
        
        plt.title(title)
        plt.xlabel('鏃堕棿姝?)
        plt.ylabel('鍊?)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save_models(self, traffic_model_path='traffic_model.h5', energy_model_path='energy_model.h5', link_quality_model_path='link_quality_model.h5'):
        """淇濆瓨璁粌濂界殑妯″瀷
        
        鍙傛暟:
            traffic_model_path: 娴侀噺妯″瀷淇濆瓨璺緞
            energy_model_path: 鑳介噺妯″瀷淇濆瓨璺緞
            link_quality_model_path: 閾捐矾璐ㄩ噺妯″瀷淇濆瓨璺緞
        """
        if self.is_trained_traffic:
            self.traffic_model.save(traffic_model_path)
            print(f"[SAVE] Traffic model saved to {traffic_model_path}")
        
        if self.is_trained_energy:
            self.energy_model.save(energy_model_path)
            print(f"[SAVE] Energy model saved to {energy_model_path}")
            
        if self.is_trained_link_quality:
            self.link_quality_model.save(link_quality_model_path)
            print(f"[SAVE] Link-quality model saved to {link_quality_model_path}")
        
        # 淇濆瓨妯″瀷鐗堟湰鍜屾洿鏂版椂闂?        from datetime import datetime
        self.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] Model version: {self.model_version}, last update: {self.last_update}")
    
    def load_models(self, traffic_model_path='traffic_model.h5', energy_model_path='energy_model.h5', link_quality_model_path='link_quality_model.h5'):
        """鍔犺浇棰勮缁冩ā鍨?        
        鍙傛暟:
            traffic_model_path: 娴侀噺妯″瀷璺緞
            energy_model_path: 鑳介噺妯″瀷璺緞
            link_quality_model_path: 閾捐矾璐ㄩ噺妯″瀷璺緞
        """
        try:
            self.traffic_model = tf.keras.models.load_model(traffic_model_path)
            self.is_trained_traffic = True
            print(f"[LOAD] Traffic model loaded: {traffic_model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load traffic model: {e}")
        
        try:
            self.energy_model = tf.keras.models.load_model(energy_model_path)
            self.is_trained_energy = True
            print(f"[LOAD] Energy model loaded: {energy_model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load energy model: {e}")
            
        try:
            self.link_quality_model = tf.keras.models.load_model(link_quality_model_path)
            self.is_trained_link_quality = True
            print(f"[LOAD] Link-quality model loaded: {link_quality_model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load link-quality model: {e}")
            
        # 鏇存柊鍔犺浇鏃堕棿
        from datetime import datetime
        self.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] Models loaded successfully. Current time: {self.last_update}")

    def integrated_prediction_for_routing(self, node_id, neighbor_nodes, recent_data):
        """涓篍EHFR鍗忚鐨勮矾鐢卞喅绛栨彁渚涢泦鎴愰娴嬬粨鏋?        
        鍙傛暟:
            node_id: 褰撳墠鑺傜偣ID
            neighbor_nodes: 閭诲眳鑺傜偣鍒楄〃锛屾瘡涓厓绱犱负鍖呭惈鑺傜偣淇℃伅鐨勫瓧鍏?            recent_data: 鍖呭惈鍚勭鏈€杩戞暟鎹殑瀛楀吀锛屾牸寮忎负 {data_type: data_array}
            
        杩斿洖:
            routing_metrics: 璺敱鍐崇瓥鎸囨爣瀛楀吀锛屽寘鍚悇绉嶉娴嬬粨鏋滃拰寤鸿
        """
        # 鍒濆鍖栬矾鐢辨寚鏍囧瓧鍏?        routing_metrics = {
            'node_id': node_id,
            'timestamp': None,
            'predictions': {},
            'recommendations': {}
        }
        
        # 璁板綍褰撳墠鏃堕棿
        from datetime import datetime
        routing_metrics['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 鎻愬彇鍚勭鏁版嵁
        recent_traffic = recent_data.get('traffic', None)
        recent_energy = recent_data.get('energy', None)
        recent_link_quality = recent_data.get('link_quality', None)
        
        # 鍒涘缓澶氱壒寰佽緭鍏ュ瓧鍏?        multi_features = {k: v for k, v in recent_data.items() if k not in ['traffic', 'energy', 'link_quality']}
        
        # 棰勬祴缁撴灉瀛楀吀
        predictions = {}
        
        # 1. 娴侀噺棰勬祴
        if recent_traffic is not None and self.is_trained_traffic:
            try:
                future_traffic = self.predict_traffic(recent_traffic, multi_features if self.multi_feature else None)
                predictions['traffic'] = future_traffic.tolist()
                
                # 妫€娴嬫嫢濉?                capacity_threshold = recent_data.get('capacity_threshold', np.max(recent_traffic) * 1.5)
                congestion_predicted, time_to_congestion, congestion_probability = self.predict_congestion(
                    recent_traffic, capacity_threshold, multi_features=multi_features if self.multi_feature else None
                )
                predictions['congestion'] = {
                    'predicted': congestion_predicted,
                    'time_to_congestion': time_to_congestion,
                    'probability': float(congestion_probability)
                }
            except Exception as e:
                predictions['traffic_error'] = str(e)
        
        # 2. 鑳介噺棰勬祴
        if recent_energy is not None and self.is_trained_energy:
            try:
                future_energy = self.predict_energy(recent_energy, multi_features if self.multi_feature else None)
                predictions['energy'] = future_energy.tolist()
                
                # 妫€娴嬭妭鐐规晠闅?                energy_threshold = recent_data.get('energy_threshold', np.min(recent_energy) * 0.5)
                additional_factors = {
                    'traffic_congestion': 0.3,
                    'link_quality': 0.3,
                    'temperature': 0.2
                }
                if 'congestion_probability' in predictions.get('congestion', {}):
                    additional_factors['congestion_probability'] = predictions['congestion']['probability']
                if recent_link_quality is not None:
                    additional_factors['link_quality_value'] = np.mean(recent_link_quality)
                if 'temperature' in recent_data:
                    additional_factors['temperature_value'] = np.mean(recent_data['temperature'])
                    additional_factors['temperature_threshold'] = recent_data.get('temperature_threshold', 70)
                
                failure_predicted, time_to_failure, failure_probability = self.predict_node_failure(
                    recent_energy, energy_threshold, 
                    multi_features=multi_features if self.multi_feature else None,
                    additional_factors=additional_factors
                )
                predictions['node_failure'] = {
                    'predicted': failure_predicted,
                    'time_to_failure': time_to_failure,
                    'probability': float(failure_probability)
                }
            except Exception as e:
                predictions['energy_error'] = str(e)
        
        # 3. 閾捐矾璐ㄩ噺棰勬祴
        if recent_link_quality is not None and self.is_trained_link_quality:
            try:
                future_link_quality = self.predict_link_quality(recent_link_quality, multi_features if self.multi_feature else None)
                predictions['link_quality'] = future_link_quality.tolist()
            except Exception as e:
                predictions['link_quality_error'] = str(e)
        
        # 灏嗛娴嬬粨鏋滄坊鍔犲埌璺敱鎸囨爣涓?        routing_metrics['predictions'] = predictions
        
        # 鐢熸垚璺敱寤鸿
        recommendations = {}
        
        # 1. 鑺傜偣鎺掑悕 - 鍩轰簬棰勬祴缁撴灉涓洪偦灞呰妭鐐硅瘎鍒?        if neighbor_nodes:
            node_scores = {}
            for node in neighbor_nodes:
                # 鍒濆鍒嗘暟
                score = 50.0
                
                # 鏍规嵁鑳介噺姘村钩璋冩暣鍒嗘暟
                if 'energy_level' in node:
                    score += node['energy_level'] * 20  # 鑳介噺姘村钩鑼冨洿鍋囪涓篬0,1]
                
                # 鏍规嵁閾捐矾璐ㄩ噺璋冩暣鍒嗘暟
                if 'link_quality' in node:
                    score += node['link_quality'] * 15  # 閾捐矾璐ㄩ噺鑼冨洿鍋囪涓篬0,1]
                
                # 鏍规嵁鎷ュ鎯呭喌璋冩暣鍒嗘暟
                if 'congestion_level' in node:
                    score -= node['congestion_level'] * 25  # 鎷ュ姘村钩鑼冨洿鍋囪涓篬0,1]
                
                # 鏍规嵁鏁呴殰姒傜巼璋冩暣鍒嗘暟
                if 'failure_probability' in node:
                    score -= node['failure_probability'] * 30  # 鏁呴殰姒傜巼鑼冨洿涓篬0,1]
                
                # 瀛樺偍鑺傜偣鍒嗘暟
                node_scores[node['id']] = max(0, min(100, score))  # 纭繚鍒嗘暟鍦╗0,100]鑼冨洿鍐?            
            # 瀵硅妭鐐硅繘琛屾帓鍚?            ranked_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations['node_ranking'] = ranked_nodes
        
        # 2. 璺敱绛栫暐寤鸿
        routing_strategy = {}
        
        # 鍩轰簬鎷ュ棰勬祴鐨勭瓥鐣?        if 'congestion' in predictions and predictions['congestion']['predicted']:
            routing_strategy['congestion_avoidance'] = True
            routing_strategy['load_balancing'] = True
        else:
            routing_strategy['congestion_avoidance'] = False
            routing_strategy['load_balancing'] = False
        
        # 鍩轰簬鑺傜偣鏁呴殰棰勬祴鐨勭瓥鐣?        if 'node_failure' in predictions and predictions['node_failure']['predicted']:
            routing_strategy['fault_tolerance'] = True
            routing_strategy['redundant_paths'] = True
        else:
            routing_strategy['fault_tolerance'] = False
            routing_strategy['redundant_paths'] = False
        
        # 鍩轰簬閾捐矾璐ㄩ噺棰勬祴鐨勭瓥鐣?        if 'link_quality' in predictions:
            avg_future_link_quality = np.mean(predictions['link_quality'])
            if avg_future_link_quality < 0.5:  # 鍋囪閾捐矾璐ㄩ噺鑼冨洿涓篬0,1]
                routing_strategy['link_quality_aware'] = True
                routing_strategy['retransmission'] = True
            else:
                routing_strategy['link_quality_aware'] = False
                routing_strategy['retransmission'] = False
        
        # 娣诲姞璺敱绛栫暐鍒板缓璁腑
        recommendations['routing_strategy'] = routing_strategy
        
        # 3. 鑳介噺绠＄悊寤鸿
        energy_management = {}
        
        # 鍩轰簬鑳介噺棰勬祴鐨勭瓥鐣?        if 'energy' in predictions:
            min_future_energy = np.min(predictions['energy'])
            if min_future_energy < 20:  # 鍋囪鑳介噺闃堝€间负20
                energy_management['energy_conservation'] = True
                energy_management['duty_cycling'] = True
                energy_management['transmission_power_control'] = True
            else:
                energy_management['energy_conservation'] = False
                energy_management['duty_cycling'] = False
                energy_management['transmission_power_control'] = False
        
        # 娣诲姞鑳介噺绠＄悊绛栫暐鍒板缓璁腑
        recommendations['energy_management'] = energy_management
        
        # 灏嗗缓璁坊鍔犲埌璺敱鎸囨爣涓?        routing_metrics['recommendations'] = recommendations
        
        return routing_metrics

# 娴嬭瘯浠ｇ爜
if __name__ == "__main__":
    # 鐢熸垚妯℃嫙鏁版嵁
    np.random.seed(42)
    
    # 妯℃嫙缃戠粶娴侀噺鏁版嵁锛堝甫鏈夊懆鏈熸€у拰瓒嬪娍锛?    time_steps = 200
    t = np.arange(time_steps)
    trend = 0.01 * t
    seasonality = 10 * np.sin(t * (2 * np.pi / 24))  # 24灏忔椂鍛ㄦ湡
    noise = 2 * np.random.normal(0, 1, time_steps)
    traffic_data = trend + seasonality + noise
    
    # 妯℃嫙鑺傜偣鑳介噺鏁版嵁锛堥€掑噺瓒嬪娍锛?    initial_energy = 100
    energy_decay = np.exp(-0.01 * t)
    energy_noise = 2 * np.random.normal(0, 1, time_steps)
    energy_data = initial_energy * energy_decay + energy_noise
    energy_data = np.clip(energy_data, 0, initial_energy)  # 纭繚鑳介噺闈炶礋涓斾笉瓒呰繃鍒濆鍊?    
    # 妯℃嫙閾捐矾璐ㄩ噺鏁版嵁锛堟尝鍔ㄨ秼鍔匡級
    link_quality_base = 0.8 - 0.3 * (t / time_steps)  # 鍩虹閾捐矾璐ㄩ噺闅忔椂闂翠笅闄?    link_quality_noise = 0.1 * np.random.normal(0, 1, time_steps)
    link_quality_data = link_quality_base + link_quality_noise
    link_quality_data = np.clip(link_quality_data, 0, 1)  # 纭繚閾捐矾璐ㄩ噺鍦╗0,1]鑼冨洿鍐?    
    # 鍒涘缓LSTM棰勬祴妯″瀷
    sequence_length = 24  # 浣跨敤24涓椂闂存鐨勫巻鍙叉暟鎹?    prediction_horizon = 12  # 棰勬祴鏈潵12涓椂闂存
    
    lstm_predictor = LSTMPrediction(sequence_length, prediction_horizon, multi_feature=True)
    
    # 璁粌娴侀噺妯″瀷锛堜娇鐢ㄥ墠160涓暟鎹偣锛?    traffic_train = traffic_data[:160]
    history_traffic = lstm_predictor.train_traffic_model(traffic_train, epochs=50)
    
    # 璁粌鑳介噺妯″瀷
    energy_train = energy_data[:160]
    history_energy = lstm_predictor.train_energy_model(energy_train, epochs=50)
    
    # 璁粌閾捐矾璐ㄩ噺妯″瀷
    link_quality_train = link_quality_data[:160]
    history_link_quality = lstm_predictor.train_link_quality_model(link_quality_train, epochs=50)
    
    # 娴嬭瘯娴侀噺棰勬祴
    test_start = 160
    recent_traffic = traffic_data[test_start:test_start+sequence_length]
    predicted_traffic = lstm_predictor.predict_traffic(recent_traffic)
    actual_future_traffic = traffic_data[test_start+sequence_length:test_start+sequence_length+prediction_horizon]
    
    print("[RESULT] Traffic prediction:")
    print("Predicted:", predicted_traffic)
    print("Actual:", actual_future_traffic)
    
    # 娴嬭瘯鑳介噺棰勬祴
    recent_energy = energy_data[test_start:test_start+sequence_length]
    predicted_energy = lstm_predictor.predict_energy(recent_energy)
    actual_future_energy = energy_data[test_start+sequence_length:test_start+sequence_length+prediction_horizon]
    
    print("\n[RESULT] Energy prediction:")
    print("Predicted:", predicted_energy)
    print("Actual:", actual_future_energy)
    
    # 娴嬭瘯閾捐矾璐ㄩ噺棰勬祴
    recent_link_quality = link_quality_data[test_start:test_start+sequence_length]
    predicted_link_quality = lstm_predictor.predict_link_quality(recent_link_quality)
    actual_future_link_quality = link_quality_data[test_start+sequence_length:test_start+sequence_length+prediction_horizon]
    
    print("\n[RESULT] Link-quality prediction:")
    print("Predicted:", predicted_link_quality)
    print("Actual:", actual_future_link_quality)
    
    # 娴嬭瘯鑺傜偣鏁呴殰棰勬祴
    failure_predicted, time_to_failure, failure_probability = lstm_predictor.predict_node_failure(recent_energy, energy_threshold=20)
    print(f"\n[RESULT] Node failure prediction: {failure_predicted}")
    if failure_predicted:
        print(f"Estimated failure in {time_to_failure} time steps")
    print(f"Failure probability: {failure_probability:.2f}")
    
    # 娴嬭瘯缃戠粶鎷ュ棰勬祴
    congestion_predicted, time_to_congestion, congestion_probability = lstm_predictor.predict_congestion(recent_traffic, capacity_threshold=25)
    print(f"\n[RESULT] Network congestion prediction: {congestion_predicted}")
    if congestion_predicted:
        print(f"Estimated congestion in {time_to_congestion} time steps")
    print(f"Congestion probability: {congestion_probability:.2f}")
    
    # 娴嬭瘯澶氱壒寰侀娴?    multi_features = {
        'temperature': np.random.normal(25, 5, sequence_length),
        'humidity': np.random.normal(60, 10, sequence_length),
        'packet_loss': np.random.uniform(0, 0.2, sequence_length)
    }
    
    # 娴嬭瘯闆嗘垚棰勬祴
    node_id = 'node_1'
    neighbor_nodes = [
        {'id': 'node_2', 'energy_level': 0.8, 'link_quality': 0.9, 'congestion_level': 0.1, 'failure_probability': 0.05},
        {'id': 'node_3', 'energy_level': 0.6, 'link_quality': 0.7, 'congestion_level': 0.3, 'failure_probability': 0.15},
        {'id': 'node_4', 'energy_level': 0.4, 'link_quality': 0.5, 'congestion_level': 0.6, 'failure_probability': 0.35}
    ]
    
    recent_data = {
        'traffic': recent_traffic,
        'energy': recent_energy,
        'link_quality': recent_link_quality,
        'temperature': multi_features['temperature'],
        'humidity': multi_features['humidity'],
        'packet_loss': multi_features['packet_loss'],
        'capacity_threshold': 25,
        'energy_threshold': 20,
        'temperature_threshold': 35
    }
    
    routing_metrics = lstm_predictor.integrated_prediction_for_routing(node_id, neighbor_nodes, recent_data)
    print("\n[SUMMARY] Integrated prediction result:")
    print(f"Node ranking: {routing_metrics['recommendations']['node_ranking']}")
    print(f"Routing strategy: {routing_metrics['recommendations']['routing_strategy']}")
    print(f"Energy management: {routing_metrics['recommendations']['energy_management']}")
    
    # 鍙鍖栭娴嬬粨鏋?    lstm_predictor.visualize_prediction(
        actual_future_traffic, recent_traffic, predicted_traffic, 
        title="Network traffic prediction"
    )
    
    lstm_predictor.visualize_prediction(
        actual_future_energy, recent_energy, predicted_energy, 
        title="鑺傜偣鑳介噺棰勬祴"
    )
    
    lstm_predictor.visualize_prediction(
        actual_future_link_quality, recent_link_quality, predicted_link_quality, 
        title="閾捐矾璐ㄩ噺棰勬祴"
    )

