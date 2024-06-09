from __future__ import annotations
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from datetime import datetime, timedelta
from typing import Optional, Union, Iterable, Sequence, Any


class Client:
    seniority: int
    home: int
    time: int
    age: int
    marital: int
    records: int
    job: int
    expenses: int
    income: int
    assets: int
    debt: int
    amount: int
    price: int

    def __init__(self, seniority: int, home: int, time: int, age: int, marital: int, records: int, job: int,
                 expenses: int, income: int, assets: int, debt: int, amount: int, price: int) -> None:
        self.seniority = seniority
        self.home = home
        self.time = time
        self.age = age
        self.marital = marital
        self.records = records
        self.job = job
        self.expenses = expenses
        self.income = income
        self.assets = assets
        self.debt = debt
        self.amount = amount
        self.price = price


class KnownClient(Client):
    status: int

    def __init__(self, status: int, seniority: int, home: int, time: int, age: int, marital: int, records: int, job: int,
                 expenses: int, income: int, assets: int, debt: int, amount: int, price: int) -> None:
        super().__init__(seniority, home, time, age, marital, records, job, expenses, income, assets, debt, amount, price)
        self.status = status


class TestingKnownClient(KnownClient):
    classification: Optional[int]

    def __init__(self, status: int, seniority: int, home: int, time: int, age: int, marital: int, records: int, job: int,
                 expenses: int, income: int, assets: int, debt: int, amount: int, price: int, classification: Optional[int] = None) -> None:
        super().__init__(status, seniority, home, time, age, marital, records, job, expenses, income, assets, debt, amount, price)
        self.classification = classification

    def is_matches(self) -> bool:
        return self.status == self.classification


class ClassifiedClient(Client):
    classification: Optional[int]

    def __init__(self, classification: Optional[int], client: UnknownClient) -> None:
        super().__init__(
            seniority=client.seniority,
            home=client.home,
            time=client.time,
            age=client.age,
            marital=client.marital,
            records=client.records,
            job=client.job,
            expenses=client.expenses,
            income=client.income,
            assets=client.assets,
            debt=client.debt,
            amount=client.amount,
            price=client.price,
        )
        self.classification = classification


class TrainingKnownClient(KnownClient):
    pass


class UnknownClient(Client):
    pass


class TrainingData:
    name: str

    uploaded: datetime
    tested: datetime

    training: list[Client]
    testing: list[Client]
    tuning: list[Hyperparameter]

    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: datetime
        self.tested: datetime
        self.training: list[TrainingKnownClient] = []
        self.testing: list[TestingKnownClient] = []
        self.tuning: list[Hyperparameter] = []

    def load(self, raw_data: Iterable[dict[str, str]]) -> None:
        for count, data in enumerate(raw_data):
            if count % 5 == 0:
                testing_client = TestingKnownClient(
                    seniority=int(data["seniority"]),
                    home=int(data["home"]),
                    time=int(data["time"]),
                    age=int(data["age"]),
                    marital=int(data["marital"]),
                    records=int(data["records"]),
                    job=int(data["job"]),
                    expenses=int(data["expenses"]),
                    income=int(data["income"]),
                    assets=int(data["assets"]),
                    debt=int(data["debt"]),
                    amount=int(data["amount"]),
                    price=int(data["price"]),
                    status=int(data["status"])
                )
                self.testing.append(testing_client)
            else:
                training_data = TrainingKnownClient(
                    seniority=int(data["seniority"]),
                    home=int(data["home"]),
                    time=int(data["time"]),
                    age=int(data["age"]),
                    marital=int(data["marital"]),
                    records=int(data["records"]),
                    job=int(data["job"]),
                    expenses=int(data["expenses"]),
                    income=int(data["income"]),
                    assets=int(data["assets"]),
                    debt=int(data["debt"]),
                    amount=int(data["amount"]),
                    price=int(data["price"]),
                    status=int(data["status"]),
                )
                self.training.append(training_data)
        self.uploaded = datetime.utcnow() + timedelta(hours=3)

    def test(self, parameter: Hyperparameter) -> None:
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.utcnow() + timedelta(hours=3)

    @staticmethod
    def classify(parameter: Hyperparameter, client: UnknownClient) -> ClassifiedClient:
        return ClassifiedClient(
            classification=parameter.classify_list([client])[0], client=client
        )

    @staticmethod
    def get_clients_list(clients: Sequence[Client]) -> list[list[int]]:
        return [
            [
                client.seniority,
                client.home,
                client.age,
                client.marital,
                client.records,
                client.expenses,
                client.assets,
                client.amount,
                client.price,
            ]
            for client in clients
        ]

    @staticmethod
    def get_statuses(clients: Sequence[KnownClient]) -> list[int]:
        return [client.status for client in clients]


class Hyperparameter:
    data: Optional[TrainingData]
    max_depth: int
    min_samples_leaf: int
    quality: float

    def __init__(self, max_depth: int, min_samples_leaf: int, training: TrainingData) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.data = training
        self.quality: float

    def test(self) -> None:
        training_data = self.data
        if not training_data:
            raise RuntimeError("No training data")
        test_data = training_data.testing
        y_test = TrainingData.get_statuses(test_data)
        y_predict = self.classify_list(test_data)
        self.quality = roc_auc_score(y_test, y_predict)
        for i in range(len(y_predict)):
            test_data[i].classification = y_predict[i]

    def classify_list(self, clients: Sequence[Union[UnknownClient, TestingKnownClient]]) -> list[Any]:
        training_data = self.data
        if not training_data:
            raise RuntimeError("No training object")
        x_predict = TrainingData.get_clients_list(clients)
        x_train = TrainingData.get_clients_list(training_data.training)
        y_train = TrainingData.get_statuses(training_data.training)

        classifier = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        classifier = classifier.fit(x_train, y_train)
        y_predict = classifier.predict(x_predict).tolist()
        return [y_predict]
