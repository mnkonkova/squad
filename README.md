# squad
## О проекте
Цель проекта была в получении возможности по представленному абзацу и вопросу находить часть непрерывного текста в абзаце в качестве ответа на вопрос.

Данная обученная модель способна предсказывать ответ с точностью ~30 +- 3% в метрике F1 score.  
## Требования
1) Python 3.*
2) pip для python3
3) пакеты numpy, msgpack, gensim, tensorflow
## Инструкция
1) Загрузите [embedings](https://www.dropbox.com/s/r33ljlagys0wscb/data.msgpack?dl=1) в папку со скриптами
2) Загрузите [данные](https://www.dropbox.com/s/83txkgiqmdlv1m3/meta.msgpack?dl=1) в папку со скриптами
3) Загрузите [data модели](https://www.dropbox.com/s/znxjvlwih4cf9io/my_test_model_1.data-00000-of-00001?dl=1) в ту же папку
4) Остальные файлы для predict так же поместите в ту же папку
## Применение 
1) Чтобы обучить собственную модель: 

  a) создайте папку для сохранения модели ``` mkdir data && mkdir data/model_2```

  b) запустите ```python3 model.py``` 

2) Чтобы посчитать F1 score на выборке, запустите ```python3 test.py```
3) Чтобы задать вопрос и получить ответ, запустите ```python3 demo.py```
