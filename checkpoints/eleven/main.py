import pandas as pd
import numpy as np
from models import TrainingData, Hyperparameter, UnknownClient
from sklearn.model_selection import train_test_split

df = pd.read_csv('../../data.csv')
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

uc = UnknownClient(
    seniority=td.training[0].seniority,
    home=td.training[0].home,
    time=td.training[0].time,
    age= td.training[0].age,
    marital=td.training[0].marital,
    records=td.training[0].records,
    job=td.training[0].job,
    expenses=td.training[0].expenses,
    income=td.training[0].income,
    assets=td.training[0].assets,
    debt=td.training[0].debt,
    amount=td.training[0].amount,
    price=td.training[0].price,
)

c = td.classify(h, uc)

result = c.classification[0]

print("Result: " + str(result))

# TEST

waiting_result = 2

print("Waiting Result: " + str(waiting_result))

assert waiting_result == result

print("Test passed.")

