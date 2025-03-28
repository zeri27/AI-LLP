import os

import edge_tts
import asyncio
import pygame

gen_text = 0

async def tts(text):
    global gen_text
    tts = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural", rate='+15%', volume='-35%', pitch='-30Hz')
    await tts.save(f"output{gen_text}.mp3")

    pygame.mixer.init()
    pygame.mixer.music.load(f"output{gen_text}.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        await asyncio.sleep(1)

    gen_text += 1

#text = "Hi, 今天我们要学习一些中文. In Chinese, '你好' means 'hello'. It's a common greeting you can use when meeting people. For example, when you greet someone in the morning, you can say '早上好', which means 'Good morning'. Another useful phrase is '谢谢', which means 'thank you'. If you want to say 'You're welcome', you can reply with '不客气'. When asking someone how they are, you can say '你好吗?', which translates to 'How are you?'. A typical response would be '我很好', meaning 'I’m good.' In a restaurant, you can use '菜单' for 'menu' and '水' for 'water'. If you want to say 'The food is delicious', you can say '这道菜很好吃'."
#asyncio.run(tts(text))