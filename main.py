import logging
import os


from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, BotCommand
import nest_asyncio
import asyncio



from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from handlers.user_private import user_private_router




# Конфигурация логирования.
logging.basicConfig(level=logging.INFO)

# Инициализируем бота и диспетчера.
bot = Bot(token=os.getenv('TOKEN'))
dp = Dispatcher()
dp.include_router(user_private_router)




# Регистрируем асинхронную функцию в диспетчере,
# которая будет выполняться на старте бота,
nest_asyncio.apply()
async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())