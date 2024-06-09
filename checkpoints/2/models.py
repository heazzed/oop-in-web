from __future__ import annotations
from typing import Optional, Iterable, Sequence, Any
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeClassifier


class Client:
    status: Optional[int]
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
                 expenses: int, income: int, assets: int, debt: int, amount: int, price: int, status: Optional[int]) -> None:
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
        self.status = status
        self.classification: Optional[str] = None

    def set_classification(self, classification: str) -> None:
        self.classification = classification

    def is_matches(self) -> bool:
        return self.status == self.classification


class Hyperparameter:
    max_depth: int
    min_samples_leaf: int
    data: Optional[TrainingData]
    quality: float

    def __init__(self, max_depth: int, min_samples_leaf: int, training: TrainingData) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.data = training

    def test(self) -> None:
        pass_count, fail_count = 0, 0
        if self.data is not None:
            for client in self.data.testing:
                if client.is_matches():
                    pass_count += 1
                else:
                    fail_count += 1
        self.quality = pass_count / (pass_count + fail_count)

    def classify(self, clients: Sequence[Client]) -> list[Any]:
        training_data = self.data
        if not training_data:
            raise RuntimeError("No training object")

        x_predict = self.get_clients_list(clients)

        x_train = self.get_clients_list(training_data.training)
        y_train = [client.status for client in training_data.training]

        classifier = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        classifier = classifier.fit(x_train, y_train)
        y_predict = classifier.predict(x_predict).tolist()
        return [y_predict]

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


class TrainingData:
    name: str

    uploaded: datetime
    tested: datetime

    training: list[Client] = list()
    testing: list[Client] = list()
    tuning: list[Hyperparameter] = list()

    def __init__(self, name: str) -> None:
        self.name = name

    def load(self, raw_data: Iterable[dict[str, int]]) -> None:
        for count, data in enumerate(raw_data):
            client = Client(
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

            if count % 5 == 0:
                self.testing.append(client)
            else:
                self.training.append(client)

        self.uploaded = datetime.utcnow() + timedelta(hours=3)

    def test(self, parameter: Hyperparameter) -> None:
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.utcnow() + timedelta(hours=3)

    @staticmethod
    def classify(parameter: Hyperparameter, client: Client) -> Client:
        classification = parameter.classify([client])[0]
        client.set_classification(classification)
        return client
