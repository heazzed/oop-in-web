import pandas as pd
import numpy as np
from models import TrainingData, Hyperparameter, UnknownClient
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

df = pd.read_csv('../../data.csv')
df.columns = df.columns.str.lower()

status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}

home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}

# df.home = df.home.map(home_values)
# df.status = df.status.map(status_values)
# df.marital = df.marital.map(marital_values)
# df.records = df.records.map(records_values)
# df.job = df.job.map(job_values)

df.head()
df.describe().round()

for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999, value=np.nan)

df = df[df['status'] != 'unk']

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=11)

y_train = (df_train.status == 'default').values
y_val = (df_val.status == 'default').values

#del df_train['status']
del df_val['status']

df_train = df_train.fillna(0)
df_val = df_val.fillna(0)

dict_train = df_train.to_dict(orient='records')
dict_val = df_val.to_dict(orient='records')

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(dict_train)
X_val = dv.transform(dict_val)

td = TrainingData('test')
td.load(dict_train)

depth = 5
leaf = 5

h = Hyperparameter(depth, leaf, td)

uc = UnknownClient(
    seniority=td.testing[0].seniority,
    home=td.testing[0].home,
    time=td.testing[0].time,
    age= td.testing[0].age,
    marital=td.testing[0].marital,
    records=td.testing[0].records,
    job=td.testing[0].job,
    expenses=td.testing[0].expenses,
    income=td.testing[0].income,
    assets=td.testing[0].assets,
    debt=td.testing[0].debt,
    amount=td.testing[0].amount,
    price=td.testing[0].price,
)

c = td.classify(h, uc)

result = c.classification[0]

print("Result: " + str(result))

# TEST

waiting_result = 1

print("Waiting Result: " + str(waiting_result))

assert waiting_result == result

print("Test passed.")
