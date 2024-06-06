from __future__ import annotations
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from datetime import datetime, timedelta
from typing import Optional, Union, Iterable, Sequence, Any, cast


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

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> KnownClient:
        if data["status"] not in {1, 2, 0}:
            raise InvalidClientError(f"Invalid client in {data!r}")
        try:
            return cls(
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
        except ValueError as exception:
            raise InvalidClientError(f"Invalid {data!r} \n {exception}")


class TrainingKnownClient(KnownClient):

    @classmethod
    def from_dict(cls, row: dict[str, int]) -> TrainingKnownClient:
        return cast(TrainingKnownClient, super().from_dict(row))


class TestingKnownClient(KnownClient):
    classification: Optional[str]

    def __init__(self, status: int, seniority: int, home: int, time: int, age: int, marital: int, records: int, job: int,
                 expenses: int, income: int, assets: int, debt: int, amount: int, price: int, classification: Optional[str] = None) -> None:
        super().__init__(status, seniority, home, time, age, marital, records, job, expenses, income, assets, debt, amount, price)
        self.classification = classification

    def is_matches(self) -> bool:
        return self.status == self.classification

    @classmethod
    def from_dict(cls, row: dict[str, int]) -> TestingKnownClient:
        return cast(TestingKnownClient, super().from_dict(row))


class UnknownClient(Client):
    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "UnknownClient":
        if set(data.keys()) != {"seniority", "home", "time", "age", "marital", "records", "job",
                                "expenses", "income", "assets", "debt", "amount", "price"}:
            raise InvalidClientError(f"Invalid fields in {data!r}")
        try:
            return cls(
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
                price=int(data["price"])
            )
        except (KeyError, ValueError):
            raise InvalidClientError(f"invalid {data!r}")


class ClassifiedClient(Client):
    classification: Optional[str]

    def __init__(self, classification: Optional[str], client: UnknownClient) -> None:
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
            price=client.price
        )
        self.classification = classification


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
        training_data: Optional[TrainingData] = self.data
        if not training_data:
            raise RuntimeError("No training data")
        test_data = training_data.testing
        y_test = TrainingData.get_statuses_clients(test_data)
        y_predict = self.classify_list(test_data)
        self.quality = roc_auc_score(y_test, y_predict)
        for i in range(len(y_predict)):
            test_data[i].classification = y_predict[i]

    def classify_list(self, clients: Sequence[Union[UnknownClient, TestingKnownClient]]) -> list[Any]:
        training_data = self.data
        if not training_data:
            raise RuntimeError("No training object")
        x_predict = TrainingData.get_list_clients(clients)
        x_train = TrainingData.get_list_clients(training_data.training)
        y_train = TrainingData.get_statuses_clients(training_data.training)

        classifier = DecisionTreeClassifier()
        classifier = classifier.fit(x_train, y_train)
        y_predict = classifier.predict(x_predict).tolist()
        return [y_predict]


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

    def load(self, raw_data: Iterable[dict[str, int]]) -> None:
        for n, row in enumerate(raw_data):
            if n % 5 == 0:
                testing_client = TestingKnownClient.from_dict(row)
                self.testing.append(testing_client)
            else:
                training_data = TrainingKnownClient.from_dict(row)
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
    def get_list_clients(clients: Sequence[Client]) -> list[list[int]]:
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
    def get_statuses_clients(clients: Sequence[KnownClient]) -> list[int]:
        return [client.status for client in clients]

    @staticmethod
    def get_client_as_list(client: Client) -> list[list[int]]:
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
        ]


class InvalidClientError(ValueError):
    pass
