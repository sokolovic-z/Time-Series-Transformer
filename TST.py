import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
    
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        pos_encoding = np.zeros((max_len, d_model))
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, x, training=False):
        attn_output = self.att(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        res1 = x + attn_output
        out1 = self.layernorm1(res1)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        res2 = out1 + ffn_output
        out2 = self.layernorm2(res2, training=training)

        return out2


def build_tst_model(seq_len, feature_dim, d_model=64, num_heads=4, dff=128, num_layers=2, dropout_rate=0.1):
    inputs = Input(shape=(seq_len, feature_dim), batch_size=None)
    x = Dense(d_model)(inputs) # input embedding: project feature_dim to d_model
    x = PositionalEncoding(d_model)(x) # positional encoding

    # Stacked Transformer encoder layers
    for _ in range(num_layers):
        encoder_layer = TransformerEncoder(d_model, num_heads, dff, dropout_rate)
        x = encoder_layer(x)

    # Pooling over sequence (use last time step, max, or global average)
    x = GlobalAveragePooling1D()(x)
    
    # Final output regression layer (predict next value)
    outputs = Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='model_TST')
    return model