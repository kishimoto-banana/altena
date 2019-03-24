import category_encoders as ce
import pandas as pd


class OrdinalEncoder:
    def __init__(self):

        self.columns = None
        self.encoder = None
        self.mapping = None

    def fit(self, df) -> None:
        """
        :param df: pandas.DataFrame
        """

        df.fillna('', inplace=True)
        df = df.astype('str')

        self.columns = df.columns.tolist()
        self.encoder = ce.OrdinalEncoder(
            cols=self.columns, handle_unknown='impute')
        self.encoder.fit(df)
        self.mapping = self.__get_mapping()

    def fit_transform(self, df):
        """
        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """

        df.fillna('', inplace=True)
        df = df.astype('str')

        self.columns = df.columns.tolist()
        self.encoder = ce.OrdinalEncoder(
            cols=self.columns, handle_unknown='impute')
        df_encoded = self.encoder.fit_transform(df)
        self.mapping = self.__get_mapping()
        return df_encoded

    def transform(self, df):
        """
        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """

        df.fillna('', inplace=True)
        df = df.astype('str')
        df_encoded = self.encoder.transform(df)
        return df_encoded

    def __get_mapping(self):

        mappings = list()
        for row in self.encoder.category_mapping:
            mappings.append((row['col'], 'unknown', 0))
            mappings.extend([tuple([row['col']]) + i for i in row['mapping']])
        df_ord_map = pd.DataFrame(
            mappings, columns=['column', 'original', 'encoded'])
        return df_ord_map
