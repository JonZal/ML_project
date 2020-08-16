import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statistics
from scipy import stats
from collections import Counter
from sklearn import preprocessing
from operator import itemgetter

sns.set(style="ticks", color_codes=True)
pandas.set_option('display.expand_frame_repr', False)

data = pandas.read_csv('C:\\Users\\Zalys\\Desktop\\Darbai\\breach_report.csv')
data['Breach Submission Date'] = pandas.to_datetime(data['Breach Submission Date'], format='%m/%d/%Y', errors='coerce')

pandas.options.display.float_format = "{:.2f}".format

len = Counter(data['State Population']).keys()
print(len.__len__())
len = Counter(data['Individuals Affected']).keys()
print(len.__len__())
len = Counter(data['Breach Submission Date']).keys()
print(len.__len__())

print(statistics.median(data['State Population']))
print(statistics.median(data['Individuals Affected']))
print()

perc = [.25, .50, .75]
include = 'all'

# calling describe method
desc = data.describe(percentiles=perc, include=include)
print(desc)

columns = list(data)

for i in columns:
    print(i)
    print(data[ (data[i].notnull()) & (data[i]!=u'') ].index)
    print(data[i].value_counts())
    print("")

# remove outliers
z = np.abs(stats.zscore(data['Individuals Affected']))
ind_rem = np.where(z > 3)
#print(ind_rem)
for i in ind_rem:
    data = data.drop(i)

# pašalinamos tuščios valstijos
data = data[data['State Population'] > 0]
#print(data)


# histogramos
data.groupby(pandas.Grouper(key='Breach Submission Date', freq='1M')).count().plot(kind='bar')
n, bins, patches = plt.hist(data['Individuals Affected'], bins='auto', alpha=0.7, rwidth=0.85)
plt.xlim(0, 20000)
plt.show()
n, bins, patches = plt.hist(data['State Population'], bins=10, alpha=0.7, rwidth=0.85)
plt.show()


data['Breach Submission Date'] = pandas.to_datetime(data['Breach Submission Date'], format='%m/%d/%Y', errors='coerce')
data.set_index('Breach Submission Date', inplace=True)
print(data.info())

# tolydziuju santykis
plt.scatter(data['State Population'], data['Individuals Affected'], alpha=0.3, edgecolors='k')

# kategoriniu ir kategoriniu santykis
type_by_business = pandas.crosstab(index = data['Type of Breach'], columns = data['Business Associate Present'] )
print(type_by_business)
type_by_business.plot(kind="bar", stacked=True)

type_by_business = pandas.crosstab(index = data['Type of Breach'], columns = data['Covered Entity Type'] )
print(type_by_business)
type_by_business.plot(kind="bar", stacked=True)

type_by_business = pandas.crosstab(index = data['Covered Entity Type'], columns = data['Business Associate Present'] )
print(type_by_business)
type_by_business.plot(kind="bar", stacked=True)


# kategoriniai ir tolydieji
sns.catplot(y="State Population", x="Type of Breach", kind="box", data=data);
sns.catplot(y="Individuals Affected", x="Type of Breach", kind="box", data=data);

# koreliacijos matrica
corrMatrix = data.corr()
print(corrMatrix)
sns.heatmap(corrMatrix, annot=True)
plt.show()

# kovariacijos matrica
covMatrix = pandas.DataFrame.cov(data)
print(covMatrix)
sns.heatmap(covMatrix, annot=True, fmt='g')
plt.show()

#normalizacija
data_ind_norm = preprocessing.maxabs_scale(data['Individuals Affected'])
print(data_ind_norm)

