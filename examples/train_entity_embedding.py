import pandas as pd
from altena.encoder import OrdinalEncoder
from altena.entity_embedding import EntityEmbedding, save_model


def main():

    # dataのロード
    df = pd.read_csv('examples/data/train.csv')
    y = df['Survived']
    df = df.drop('Survived', axis=1)

    # encoding
    encoder = OrdinalEncoder()
    df_encoded = encoder.fit_transform(df)
    encoder.mapping.to_csv(
        'examples/output/mapping.tsv', sep='\t', index=False)

    # パラメータの設定
    # 各embeddingレイヤのユニット数、ここでは全てのカテゴリ変数で3
    embedding_dims = [3 for i in range(len(df_encoded.columns))]
    # 中間層のユニット数（リストの長さが中間層数）
    hidden_units = [50, 30]
    # 各中間層の活性化関数（hidden_unitsのインデックスと対応）
    activations = ['relu', 'relu']
    # 最適化法
    optimizer = 'rmsprop'
    # 損失関数
    loss = 'binary_crossentropy'
    # ドロップアウト率（Embeddingレイヤのみ）
    dropout_rate = 0.2

    # EntityEmbeddingモデルのインスタンス生成
    model = EntityEmbedding(
        encoder=encoder,
        embedding_dims=embedding_dims,
        hidden_units=hidden_units,
        activations=activations,
        optimizer=optimizer,
        loss=loss,
        dropout_rate=dropout_rate)

    # 学習
    X = df_encoded.values
    model.fit(X, y, epochs=10)
    save_model('examples/output/model.pkl', model)


if __name__ == '__main__':
    main()
