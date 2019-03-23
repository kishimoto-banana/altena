import keras
from keras.models import Model
from keras.layers import Embedding, Input, Activation, Reshape, Dense, Concatenate, Dropout
import pickle

allowed_regularizers = ['l1', 'l2']


class EntityEmbeddingError(Exception):
    """EntityEmbeddingモジュールにおける例外の基底クラス"""


class LengthNotMatchedError(EntityEmbeddingError):
    """各リストの長さが一致しなかったときのエラー"""


class RegularizerNotAllowedError(EntityEmbeddingError):
    """許可された正則化項では無かったときのエラー"""


class EntityEmbedding:
    """
    EntityEmbeddingクラス
    """

    def __init__(self,
                 encoder,
                 embedding_dims,
                 hidden_units=[
                     100,
                 ],
                 activations=['relu'],
                 optimizer='rmsprop',
                 dropout_rate=0.0,
                 regularizers=None):
        """
        :param encoder: category_encoders.OrdinalEncoder
        :param embedding_dims: list
        :param hidden_units: list
        :param activations: list
        :param optimizer: str
        :param dropout_rate: float
        :param regularizers: list
        """
        try:
            mapping = encoder.mapping
            columns = mapping['column'].unique().tolist()
            if len(columns) != len(embedding_dims):
                raise LengthNotMatchedError(
                    'Not matched length of encoded columns and embed_dims')
            if regularizers is not None:
                if len(columns) != len(regularizers):
                    raise LengthNotMatchedError(
                        'Not matched length of encoded columns and regularizers'
                    )
                for regularizer in regularizers:
                    if regularizer[0] in allowed_regularizers:
                        RegularizerNotAllowedError(
                            'allowd regularizer are "l1" or "l2"')
        except EntityEmbeddingError:
            raise

        self.encoder = encoder
        self.model = self.__construct_model(
            encoder, embedding_dims, hidden_units, activations, optimizer,
            dropout_rate, regularizers)
        self.weights = None

    def __construct_model(self,
                          encoder,
                          embedding_dims,
                          hidden_units,
                          activations,
                          optimizer,
                          dropout_rate,
                          regularizers=None):
        """
        モデルを構築する
        :param encoder: category_encoders.OrdinalEncoder
        :param embedding_dims: list
        :param hidden_units: list
        :param activations: list
        :param optimizer: str
        :param dropout_rate: float
        :param regularizers: list
        :return: keras.model.Model
        """

        mapping = encoder.mapping
        columns = mapping['column'].unique().tolist()

        inputs = []
        embeds = []
        for idx, (column, embed_dim) in enumerate(
                zip(columns, embedding_dims)):
            input_cat = Input(shape=(1, ))
            input_dim = len(mapping.query('column == @column'))
            if regularizers is not None:
                regularizer = regularizers[idx]
                if regularizer[0] == 'l1':
                    embed = Embedding(
                        input_dim,
                        embed_dim,
                        input_length=1,
                        embeddings_regularizer=keras.regularizers.l1(
                            regularizer[1]))(input_cat)
                if regularizer[0] == 'l2':
                    embed = Embedding(
                        input_dim,
                        embed_dim,
                        input_length=1,
                        embeddings_regularizer=keras.regularizers.l2(
                            regularizer[1]))(input_cat)
            else:
                embed = Embedding(
                    input_dim, embed_dim, input_length=1)(input_cat)
            output = Reshape(target_shape=(embed_dim, ))(embed)
            inputs.append(input_cat)
            embeds.append(output)

        output = Concatenate()(embeds)
        output = Dropout(dropout_rate)(output)
        for hidden_unit, activation in zip(hidden_units, activations):
            output = Dense(hidden_unit, kernel_initializer='uniform')(output)
            output = Activation(activation)(output)
        output = Dense(1)(output)
        prediction = Activation('sigmoid')(output)

        model = Model(inputs=inputs, outputs=prediction)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy'])

        return model

    def fit(self, X, y, epochs=20, shuffle=True, batch_size=32):
        """
        学習を行う
        :param X: numpy.array
        :param y: numpy.array
        :param epochs: int
        :param shuffle: bool
        :param batch_size: int
        """

        self.model.fit(
            [X[:, feature_idx] for feature_idx in range(X.shape[1])],
            y,
            epochs=epochs,
            shuffle=shuffle,
            batch_size=batch_size)
        self.weights = self.get_weights()

    def predict(self, X):
        """
        予測を行う
        :param X: numpy.array
        :return: numpy.array
        """

        y_proba = self.model.predict(
            [X[:, feature_idx] for feature_idx in range(X.shape[1])])
        return y_proba

    def infer_vector(self, column, key):
        """
        特定のカラムとエンコードされたキーから分散表現を獲得する
        :param column: str
        :param key: int
        :return: numpy.array
        """

        feature_idx = self.encoder.columns.index(column)
        vector = self.weights[feature_idx][key, :]

        return vector

    def infer_vector_from_original_key(self, column, original_key):
        """
        特定のカラムとエンコード前のキーから分散表現を獲得する
        （速度面でパフォーマンスが悪いので、infer_vectorを使った方が良い）
        :param column: str
        :param original_key: str
        :return: numpy.array
        """

        encoded_key = self.encode_key(column, original_key)
        vector = self.infer_vector(column, encoded_key)

        return vector

    def encode_key(self, column, original_key):
        """
        キーをエンコードする
        :param column: str
        :param original_key: str
        :return: int
        """

        mapping = self.encoder.mapping
        is_exist = self.check_exist_key(column, original_key)
        if is_exist:
            encoded_key = mapping.query('column == @column').query(
                'original == @str(original_key)')['ord_num'].values[0]
        else:
            encoded_key = 0

        return encoded_key

    def check_exist_key(self, column, original_key):
        """

        :param column: str
        :param original_key: str
        :return: bool
        """
        mapping = self.encoder.mapping
        checked_mapping = mapping.query('column == @column').query(
            'original == @tr(original_key)')
        if len(checked_mapping) == 0:
            is_exist = False
        else:
            is_exist = True

        return is_exist

    def decode_key(self, column, key):
        """
        エンコードされたキーをデコードする
        :param column: str
        :param key: int
        :return:
        """

        mapping = self.encoder.mapping
        if key == 0:
            decoded_key = 'unknown'
        else:
            decoded_key = mapping.query('column == @column').query(
                'encoded == @str(key)')['original'].values[0]

        return decoded_key

    def get_weights(self):
        """
        学習した全分散表現を得る
        :return: np.array
        """

        return self.model.get_weights()


def save_model(filepath, model):
    """
    EntityEmbeddingモデルを保存する
    :param filepath: str
    :param model: keras.model.Model
    """

    with open(filepath, mode='wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    """
    EntityEmbeddingモデルを読み込み
    :param filepath: str
    :return: keras.model.Model
    """

    with open(filepath, mode='rb') as f:
        model = pickle.load(f)
    return model
