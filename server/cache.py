import datetime as dt


class Cache:
    def __init__(self):
        self.__cache = {}

    def __len__(self):
        return len(self.__cache)

    def __getitem__(self, item):
        return self.__cache[item]

    @property
    def expired_date(self):
        return dt.datetime.now() - dt.timedelta(hours=1)

    def is_expired(self, key):
        return self.__cache[key] < self.expired_date

    def get_expired(self):
        result = []

        for k, v in self.__cache.items():
            if self.is_expired(k):
                result.append(k)

    def add_key(self, key):
        self.__cache[key] = dt.datetime.now()

    def del_key(self, key):
        del self.__cache[key]
