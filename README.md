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
