import pandas as pd
from altena.encoder import OrdinalEncoder
from altena.entity_embedding import load_model
from sklearn.linear_model import LogisticRegression


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

    # EntityEmbeddingモデル読み込み
    model = load_model('examples/output/model.pkl')

    # 分散表現への変換
    X = model.embedding(df_encoded.values)

    # 任意の予測モデルインスタンス作成（ここではscikit-learnのロジスティック回帰）
    clf = LogisticRegression(random_state=42)

    # 学習
    clf.fit(X, y)
    print(clf.coef_)


if __name__ == '__main__':
    main()
