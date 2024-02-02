from aiogram import types, Router, Bot, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, BotCommand, FSInputFile, KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove, InputFile
import numpy as np
import torch
from PIL import Image
import gc # Интерфейс для сборщика мусора.
import os
from model import *  # Импортируем архитектуру MSGNET
from functions import *  # Импортируем функции

user_private_router = Router()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

status = {0: 'Добавь первое фото из двух', 1: 'Добавь второе фото', 2: 'Добавлено фото со стилем', 3: 'Выбери качество. Ожидай результат.'}

# Инициализируем модели и загрузим веса.
style_model = MsgNet(ngf=128).to(device)
style_model.load_state_dict(torch.load('style.model'), False)

users = {}




def transform(content_root, style_root, im_size):
    """Функция для трансформации изображения."""
    content_image = load_image_rgb(content_root, size=im_size,
                                         keep_asp=True).unsqueeze(0)
    style = load_image_rgb(style_root, size=im_size).unsqueeze(0)
    style_v = preprocess_batch(style)
    content_image = preprocess_batch(content_image)
    style_model.setTarget(style_v)
    output = style_model(content_image)
    save_image_bgr(output.data[0], 'result.jpg', False)

    # Очистка RAM.
    del content_image
    del style
    del style_v
    del output

    torch.cuda.empty_cache()
    gc.collect()


# Этот хэндлер будет срабатывать на команду "/start"
@user_private_router.message(CommandStart())
async def process_start_command(message: Message):
    """Хэндлер для команды /start"""
    await message.answer(
        text='Привет!\n\n🤗 Я бот, демонстрирующий опыт по переносу '
             'стиля с одного фото на другое фото. Отправь '
             'мне  фото (если передумал, нажми /cancel 🤔), затем '
             'отправь следующее фото, с которого хочешь перенести стиль '
             '(если опять передумал, можно так же отменить шан - нажми /cancel 🤔). '
             'После этого нужно выбрать качество изображения. Если все шаги выполнены, то '
             'осталось немного подождать и бот скоро пришлет результат. '
    )
    # Если пользователь только запустил бота и его нет в словаре '
    # 'users - добавляем его в словарь
    if message.from_user.id not in users:
        users[message.from_user.id] = {
            'flag': True,
            'content_flag': False,
            'style_flag': False,
            'final_flag': False,
            'continue_waiting': False,
            'size_img': False,
            'status': 'await for photo_1',
        }

# Этот хэндлер будет срабатывать на команду "/help"
@user_private_router.message(Command(commands='help'))
async def process_help_command(message: Message):
    """Хэндлер для команды /help"""
    await message.answer(
        text='Я бот, демонстрирующий, '
             'как работает перенос стиля '
             ' с одной картинки на другую. Выбери команду '
             'из списка ниже:\n\n'
             '/start - узнаешь, как пользоваться ботом и запустишь его.\n'
             '/cancel - отмена для выбора другого фото.\n'
             '/continue - выбор качества изображения\n'
             '/author - получить контакты создателя 😎\n'
             '/status - уточнить на каком шаге ты остановился\n'
    )


# Этот хэндлер будет срабатывать при отправке фото.
@user_private_router.message(F.photo)
async def photo_processing(message):
    """Хэндлер для реакции на отправку фото"""

    # Бот ждет первое фото.
    try:
        if users[message.from_user.id]['flag']:
            await message.bot.download(file=message.photo[-1].file_id, destination='content.jpg')
            await message.answer(text='Получена первое фото.'
                                 ' Теперь отправь фото со стилем или нажми '
                                 'команду /cancel для выбора '
                                 'другого фото.')
            users[message.from_user.id] = {
                'flag': False,
                'content_flag': True,
                'style_flag': False,
                'final_flag': False,
                'continue_waiting': False,
                'size_img': False,
                'status': 'photo_1 downloaded'}
        # Бот ожидает фото со стилем.
        else:
            await message.bot.download(file=message.photo[-1].file_id, destination='style.jpg')
            await message.answer(text='Получена второе фото. Теперь нажми команду /continue '
                                  ' или /cancel для отмены и выбора '
                                  ' фото с другим стилем 🤠. ')
            users[message.from_user.id] = {
                'flag': True,
                'content_flag': True,
                'style_flag': True,
                'final_flag': False,
                'continue_waiting': False,
                'size_img': False,
                'status': 'photo_2 downloaded'
                }
    except KeyError as e:
        await message.answer(text='Для запуска бота нажми команду из меню /start')




# Этот хэндлер будет срабатывать на команду "/cancel" для выбора другого фото или стиля.
@user_private_router.message(Command(commands='cancel'))
async def cancel_process(message: Message):
    """Хэндлер для команды /cancel"""
    try:
        if users[message.from_user.id]['content_flag'] == False and users[message.from_user.id]['style_flag'] == False:
            users[message.from_user.id] = {
                'flag': True,
                'content_flag': False,
                'style_flag': False,
                'final_flag': False,
                'continue_waiting': False,
                'size_img': False,
                'status': 'await for photo_1'
                }
            await message.answer(text="Первое фото ещё не загружено.")
        else:
            users[message.from_user.id] = {
                'flag': False,
                'content_flag': True,
                'style_flag': False,
                'final_flag': False,
                'continue_waiting': False,
                'size_img': False,
                'status': 'await for photo_2'
                }
            await message.answer(text="Отправь боту "
                                 " второе фото со стилем "
                                 " для переноса на первое фото. ")
    except KeyError as e:
        await message.answer(text='Для запуска бота нажми команду из меню /start')


@user_private_router.message(Command(commands='status'))
async def get_status(message: Message):
    """Хэндлер для команды /status"""
    try:
        if users[message.from_user.id]['content_flag'] == False:
            await message.answer(text = status[0])
        elif users[message.from_user.id]['style_flag'] == False:
            await message.answer(text = status[1])
        elif users[message.from_user.id]['continue_waiting'] == False:
            await message.answer(text = status[2])
        else:
            await message.answer(text = status[3])
    except KeyError as e:
        await message.answer(text='Для запуска бота нажми команду из меню /start')




@user_private_router.message(F.text, Command("author"))
async def creator(message: Message):
    """Хэндлер для команды /author"""
    link = 'https://github.com/Vol4uk13/Style-Transfer-Telegram-Bot'
    await message.answer(text="Бот сделан Vol4uk13 🦸."
                              "\nСсылка на код бота " + link)


# Этот хэндлер будет срабатывать на команду "/continue" для запуска процесса переноса стиля.
@user_private_router.message(F.text, Command("continue"))
async def contin(message: types.Message):
    """Хэндлер для команды /continue"""
    try:
    # Проверка на факт загрузки обеих фото.
        if users[message.from_user.id]['content_flag'] == False or users[message.from_user.id]['style_flag'] == False:
            await message.answer(text="Две фото ещё не загружены.")
        else:
        # Добавим кнопки для ответа.
            button1 = KeyboardButton(text="Low")
            button2 = KeyboardButton(text="Medium")
            button3 = KeyboardButton(text="High")
            res = ReplyKeyboardMarkup(keyboard=[[button1, button2, button3]],resize_keyboard=True,
                                      one_time_keyboard=True)

            await message.answer(text="Супер, теперь нужно выбрать качество"
                                 " для будущей картинки. Чем выше "
                                 "качество, тем медленней процесс обработки."
                                 " Если хочешь повторить пришли мне снова две фото,"
                                 " где последняя фото со стилем.", reply_markup=res)

            users[message.from_user.id] = {
                'flag': False,
                'content_flag': True,
                'style_flag': True,
                'final_flag': False,
                'continue_waiting': True,
                'size_img': False,
                'status': 'await for photo_1'}
    except KeyError as e:
        await message.answer(text='Для запуска бота нажми команду из меню /start')


@user_private_router.message(F.text == 'Low')
async def low_func(message: Message):
    """Хэндлер для качества Low"""
    users[message.from_user.id]['size_img'] = 256

    await message.answer(text='Процесс обработки запущен.Нужно подождать... ',
                         reply_markup=types.ReplyKeyboardRemove())
    transform(os.path.abspath("content.jpg"), os.path.abspath("style.jpg"), users[message.from_user.id]['size_img'])
    photo = FSInputFile(os.path.abspath("./result.jpg"))
    await  message.answer_photo(photo,caption = 'Готово!')
    users[message.from_user.id] = {
            'flag': True,
            'content_flag': False,
            'style_flag': False,
            'final_flag': False,
            'continue_waiting': False,
            'size_img': False,
            'status': 'await for photo_1'}

@user_private_router.message(F.text == 'Medium')
async def low_func(message: Message):
    """Хэндлер для качества Medium"""

    users[message.from_user.id]['size_img'] = 300

    await message.answer(text='Процесс обработки запущен.Нужно подождать... ',
                         reply_markup=types.ReplyKeyboardRemove())
    transform(os.path.abspath("content.jpg"), os.path.abspath("style.jpg"), users[message.from_user.id]['size_img'])
    photo = FSInputFile(os.path.abspath("./result.jpg"))
    await  message.answer_photo(photo,caption = 'Готово!')

    users[message.from_user.id] = {
            'flag': True,
            'content_flag': False,
            'style_flag': False,
            'final_flag': False,
            'continue_waiting': False,
            'size_img': False,
            'status': 'await for photo_1'}


@user_private_router.message(F.text == 'High')
async def low_func(message: Message):
    """Хэндлер для качества High"""

    users[message.from_user.id]['size_img'] = 350

    await message.answer(text='Процесс обработки запущен.Нужно подождать... ',
                         reply_markup=types.ReplyKeyboardRemove())
    transform(os.path.abspath("content.jpg"), os.path.abspath("style.jpg"), users[message.from_user.id]['size_img'])
    photo = FSInputFile(os.path.abspath("./result.jpg"))
    await  message.answer_photo(photo,caption = 'Готово!')

    users[message.from_user.id] = {
            'flag': True,
            'content_flag': False,
            'style_flag': False,
            'final_flag': False,
            'continue_waiting': False,
            'size_img': False,
            'status': 'await for photo_1'}
# Хэндлер для тестирования.
@user_private_router.message(F.text == 'clear')
async def clear_story(message: Message):
    """Хэндлер для очистки истории взаимодействия"""

    users[message.from_user.id] = {
            'flag': True,
            'content_flag': False,
            'style_flag': False,
            'final_flag': False,
            'continue_waiting': False,
            'size_img': False,
            'status': 'await for photo_1',
        }

    await message.answer(text='История очищена. Нажми /start для запуска бота. С тобой приятно иметь дело.😚')


# Этот хэндлер будет срабатывать на любые текстовые сообщения,
# кроме команд "/start" и "/help"
@user_private_router.message()
async def send_echo(message: Message):
    try:
        await message.answer(
            text='Я даже представить себе не могу, '
            'что ты имеешь в виду\n\n'
            'Чтобы посмотреть список доступных команд - '
            'отправь команду /help'
            )
    except KeyError as e:
        await message.answer(text='Для запуска бота нажми команду из меню /start')