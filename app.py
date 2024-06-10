import random
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from checkpoints.eleven.models import UnknownClient, TrainingData, Hyperparameter
from framework.api import API
from framework.auth import login_required, TokenMiddleware, STATIC_TOKEN, on_exception


warnings.filterwarnings("ignore", message="No directory at", module="whitenoise.base")

app = API()
users = []
data = []


class Counter:
    id = 0

    def get_id(self):
        self.id += 1
        return self.id


counter = Counter()

for u in range(5):
    c = UnknownClient(
        seniority=random.randint(0, 30),
        home=random.randint(0, 6),
        time=random.randint(1, 60),
        age=random.randint(18, 90),
        marital=random.randint(0, 5),
        records=random.randint(0, 2),
        job=random.randint(0, 4),
        expenses=random.randint(0, 9999999),
        income=random.randint(0, 9999999),
        assets=random.randint(0, 99999999),
        debt=random.randint(0, 999999),
        amount=random.randint(0, 999999),
        price=random.randint(0, 999999)
    )
    users.append({counter.get_id(): c})

df = pd.read_csv('data.csv')
df.columns = df.columns.str.lower()

for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999, value=np.nan)

df = df[df['status'] != 'unk']

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=11)

y_train = (df_train.status == 'default').values
y_val = (df_val.status == 'default').values

del df_val['status']

df_train = df_train.fillna(0)
df_val = df_val.fillna(0)

dict_train = df_train.to_dict(orient='records')

td = TrainingData('test')
td.load(dict_train)
depth = 5
leaf = 5

h = Hyperparameter(depth, leaf, td)

results = []

id = 1
for user in users:
    res = td.classify(h, user[id])
    results.append(res)
    id += 1

app.add_middleware(TokenMiddleware)
app.add_exception_handler(on_exception)


@app.route("/", allowed_methods=["get"])
def index(req, resp):
    resp.html = app.template("index.html", context={"clients": results})


@app.route("/client-list", allowed_methods=["get"])
def login(req, resp):
    text = ""
    for client in results:
        text += client.__repr__() + ";\n"
    resp.text = text


@app.route("/login", allowed_methods=["post"])
def login(req, resp):
    resp.json = {"token": STATIC_TOKEN}


@app.route("/clients", allowed_methods=["post"])
@login_required
def create_sample(req, resp):
    client = UnknownClient(seniority=req["seniority"],
                           home=req["home"],
                           time=req["time"],
                           age=req["age"],
                           marital=req["marital"],
                           records=req["records"],
                           job=req["job"],
                           expenses=req["expenses"],
                           income=req["income"],
                           assets=req["assets"],
                           debt=req["debt"],
                           amount=req["amount"],
                           price=req["price"],
                           )
    print(users)

    users.append({counter.get_id(): client})
    cc = td.classify(h, client)
    resp.status_code = 201

    resp.json = cc.__repr__()


@app.route("/clients/{id:d}", allowed_methods=["delete"])
@login_required
def delete_sample(req, resp, id):
    del users[id]

    resp.status_code = 204