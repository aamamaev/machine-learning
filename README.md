# machine-learning
 <b>0. Interactive grouping </b>   пример ООП. 

Модуль combiner.py реализует аналог блока "Interactive grouping" из SAS Miner, включая графический интерфейс.
В git выложена версия с усеченной функциональностью, позволяющая работать исключительно с категориальными признаками и заменяющая значения признаков метками групп вместо woe. Написан для python 3.6.1.

"Группинг" используется в задачах классификации. Для катеогориальных признаков "группинг" сводится к замене группы значениий признака одним. 

Все множество объектов обучающей выборки делится категориальным признаком на непересекающиеся подмножетсва, объекты каждого из подмножеств обладают своим значением признака. Для каждого подмножества можно определить долю объектов целевого класса в этом подмножетсве и доверительный интервал на эту долю. Подмножетсва с пересекающимися доверительными интервалами объединяются. Новый, производный, категориальный признак получается путем замены значений, выделяющих объединеные подмножетсва, одной меткой. 

"Группинг" напоминает построение неглубокого дерева решений на категориальном признаке с присвоением каждому листу построенного дерева отдельной метки.  

Module combiner.py implements the analog of the unit "Interactive grouping" of SAS Miner, including a graphical interface. Git posted a version with reduced functionality, which allows to work only with categorical characteristics and replacement values characteristics labels groups instead of woe. Written for python 3.6.1.

Grouping is used in classification tasks. For categorially signs "grouping" is reduced to the replacement of the group znachenii sign one.

The entire set of objects of the training sample is divided into non-intersecting subsets, the objects of each of the subsets have their own characteristic value. For each subset it is possible to determine the proportion of objects of the target class in this podmnozhestva and the confidence interval for this proportion. Subsets with overlapping confidence intervals are merged. A new, derived, categorical trait is obtained by replacing the values that allocate the merged subsets with a single label.

 <b>11. NN for Time Series </b>

Применение полносвязной нейронной сети прямого распростанения для прогнозирования большого количества (~6000 тыс.) временных рядов нагрузки на сотрудников отделений банка. Применение данного подхода позволило доститчь более высокого качетсва прогноза,  кратно сократить время обучения и примененния модели, по сравнению с подходом предполагавшем обучение SARIMAX c экзогенными признаками для каждого ряда в отдельности.  

Application of a fully connected neural network of direct propagation to predict a large number (~6000 thousand) of time series of load on employees of Bank branches. The application of this approach made it possible to achieve a higher quality of the forecast, to reduce the time of training and the application of the model, in comparison with the approach involving the training of SARIMAX with exogenous characteristics for each series separately.

 <b>1. Churn prediction </b>  смотрите пояснительную записку. 

В представленной работе описан процесс создания классификатора для конкретного  эмипирческого материала –  40 тысяч клиентов French Telecom company Orange – одного из мировых лидеров в области телекоммуникационных услуг (более 170 млн. пользователей). Рассмотрены различные методы предобработки данных и отбора значимых признаков. Оценено влияние предобработки на качество линейных методов классификации,  «случайного леса», градиентного бустинга над решающими деревьями. Опробован «stacking»-подход к решению задачи. Проведен расчет экономического эффекта от применения разработанной модели.
Программная реализация алгоритмов обработки и классификации выполнена на языке Python 2.7 в интерактивной оболочке Jupyter Notebook c использованием библиотек pandas, skipy, sklearn, seaborn. 

<b>2. Credit score </b>

Проверка различных статистических гипотез на выборке заемщиков допустивших дефолт по кредиту. 

<b>3. Time series analysis </b>

Прогнозирование ряда средней заработной платы в России.
SARIMAX

<b>4. Sentiment analysis </b>

Анализа тональности отзывов на фильмы из стандартного датасета nltk.

5. Simple clustering

PCA + DBSCAN

6. Choice of banner

На прошедшей неделе в рекламной сети параллельно размещалось два баннера. Оба баннера были показаны один миллион раз. Первый получил 10 000 кликов и 500 установок, а второй — 10 500 кликов и 440 установок. Какой баннер оставить, а какой отключить? 

7. Fraud on road

Анализ выборки страховых событий (ДТП с двумя участниками) на возможное мошенничество. Выделение тех клиентов, относительно которых существует подозрение на мошеннические действия.

8. Simple client-server

asyncio, python3

9. SQL with python

Два простых запроса. 

10. Working with logs

Имеется файл log.txt размером 1Tb, содержащий лог в следующем формате: номер записи, тип запроса, время отклика. 
Напишите на Python программу, которая для каждого типа запроса подсчитывает среднее время отклика и 95% доверительный интервал для этой величины.
