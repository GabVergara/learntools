from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from learntools.core import *


plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

base = pd.read_csv("C:\\Users\\dariu\\Desktop\\Proactiva\\data.csv", parse_dates=["fecha"],
index_col="fecha").to_period("D")
proyeccion = base[base["Equipo_fin(G)"] == "EQ. RECOVERY"]
proyeccion = proyeccion.loc[:, 'COUNT_BIG_DISTINCT(rut12)']


#ax = proyeccion.plot(**plot_params)
#ax.set(title="Cantidad asignacion")

trend = proyeccion.rolling(
window = 8,
center= True,
min_periods= 4,
).mean()

#ax = proyeccion.plot(**plot_params, alpha=0.5)
#ax = trend.plot(ax=ax, linewidth=3)


from statsmodels.tsa.deterministic import DeterministicProcess

y = proyeccion.copy()  # the target


dp = DeterministicProcess(
index= proyeccion.index,
order=1,
drop=True)


X = dp.in_sample()


X_fore = dp.out_of_sample(steps=30)

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(**plot_params, alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, linewidth=3, label="Trend", color='C0')
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color='C3')
ax.legend()
#plt.show()
print(y_fore)
