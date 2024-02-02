# Телеграм-бот для переноса стиля изображения


Финальный проект по курсу: [Deep Learning School by MIPT](https://en.dlschool.org/).


**Цель проекта:** Создать бота, которому можно отправить две фотографии и получить в ответ фото с перенесенным стилем. Так же данный бот необходимо перенсти на сервер для непрерывной работы.

Адрес бота: `[@KoshmyagBot]` (Telegram)
----------

**Нейросеть**

Для проекта выбрана нейросеть [MSG-Net](https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer) от [zhanghang1989](https://github.com/zhanghang1989), с помощью которой можно произвести стилизацию изображения. Для ускорения процесса обработки изображения использованы [веса](https://github.com/Vol4uk13/Style-Transfer-Telegram-Bot/style.model)
 от ранее обученной модели. Так же использованы дополнительные [функции](https://github.com/Vol4uk13/Style-Transfer-Telegram-Bot/functions.py) для обработки изображения. Сама модель находится [здесь](https://github.com/Vol4uk13/Style-Transfer-Telegram-Bot/model.py) 


**Бот**

Для создания телеграм-бота мною был пройден курс [Телеграм-боты на Python и AIOgram](https://stepik.org/course/120924). Бот написан с использованием библиотеки [aiogram](https://docs.aiogram.dev/en/latest/index.html). Данная библиотека использует асинхронное программирование, что позволяет боту обрабатывать несколько задач одновременно, не блокируя основной поток выполнения.

Запуск бота осуществляется через [main.py]([https://github.com/t0efL/Style-Transfer-Telegram-Bot/blob/master/](https://github.com/Vol4uk13/Style-Transfer-Telegram-Bot/main.py). Все хэндлеры(ассинхронные функции реагирования на действия пользователя) вынесены в отдельный файл [user_private.py](https://github.com/Vol4uk13/Style-Transfer-Telegram-Bot/blob/main/handlers/user_private.py) для поддержания структуры бота.

Бот создан при помощи `@BotFather`, так же в нём было настроено меню моего бота. Для запуска бота нужно использовать свой собственный токен, который выдаёт `@BotFather`.

**Deploy**

Для развертывания бота использована облачная инфраструктура сервиса Selectel с конфигурацией сервера: Ubuntu 22.04 LTS 64-bit, vCPU - 1 ядро, память - 2 ГБ. Сетевой диск: HDD Базовый на 10 гб.

![https://i.paste.pics/9FNQ5.png](https://github.com/Vol4uk13/Style-Transfer-Telegram-Bot/blob/main/images/%D0%BC%D0%B5%D0%BD%D1%8E.PNG)

**Результат**

Фото для переноса

![https://i.paste.pics/9FNQ5.png](https://github.com/Vol4uk13/Style-Transfer-Telegram-Bot/blob/main/images/photo_content.jpg)

Фото стиля.

![https://i.paste.pics/9FNQ5.png](https://github.com/Vol4uk13/Style-Transfer-Telegram-Bot/blob/main/images/photo_style.jpg)

Результат

![https://i.paste.pics/9FNQ5.png](https://github.com/Vol4uk13/Style-Transfer-Telegram-Bot/blob/main/images/photo_out.jpg)

