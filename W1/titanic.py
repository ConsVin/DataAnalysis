import pandas
data = pandas.read_csv('titanic.csv',index_col='PassengerId');

print '# 1. Number of males and females'
numMales= len(data[data.Sex=='male']);
numFemale = len(data[data.Sex=='female']);
print 'Number of  Males   %d ' % numMales
print 'Number of  Females %d ' % numFemale



print '# 2. Number of survived persons'
nSurvived = len(data[data.Survived == 1])
nTotal = len(data)
nSurvivedRatio = nSurvived / float(nTotal)*100
print 'Survived %d '  % nSurvived
print '      of %d '  % nTotal
print '   perc %.2f%%' % nSurvivedRatio

print '#3 First Class Passenger Number'
nFirstClass = len(data[data.Pclass == 1]);
print 'First Class Passengers %d' % nFirstClass 
nFirstClassRatio = nFirstClass / float(nTotal) * 100;
print 'First Class Ratio %.2f%%' % nFirstClassRatio

print '#4  Age average and median value'
fAgeMedian = data.Age.median();
fAgeMean   = data.Age.mean();
print 'Median Age %.2f' % fAgeMedian
print 'Average Age %.2f ' % fAgeMean

print '#5 Pearson Correlation'
fCorr = data.SibSp.corr(data.Parch,method='pearson');
print 'Pearson correlation between n Siblings and n Parents/Children aboard %.2f' % fCorr

print '#5 Women name most popular'
femen = data[data.Sex=='female'].Name;
notMarried = femen[femen.str.contains('Miss')];
Married = femen[femen.str.contains('Mrs')];
MarriedFirstNames = Married.str.extract('\((.*?)[\)\s]')
notMarriedFirstNames = notMarried.str.extract('Miss\.\s*(\w+)')
femenFirstNames = notMarriedFirstNames.append(MarriedFirstNames)
namesStatistic = femenFirstNames.value_counts();
print 'Top 10 names'
print namesStatistic[:10]
